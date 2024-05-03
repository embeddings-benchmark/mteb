from __future__ import annotations

import itertools
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


def evaluate_mlp(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_test: np.ndarray,
):
    scores = {}
    classifier = MLPClassifier()
    classifier.fit(embeddings_train, y_train)
    y_pred = classifier.predict(embeddings_test)
    accuracy = classifier.score(embeddings_test, y_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    scores["accuracy"] = accuracy
    scores["f1"] = f1
    lrap = label_ranking_average_precision_score(y_test, y_pred)
    scores["lrap"] = lrap
    return scores


class AbsTaskMultilabelClassification(AbsTask):
    """Abstract class for multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        label: list[Hashable]
    """

    def __init__(
        self,
        n_experiments=None,
        samples_per_label=None,
        k=3,
        batch_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        # Bootstrap parameters
        self.n_experiments = n_experiments or getattr(
            self.metadata, "n_experiments", 10
        )
        self.samples_per_label = samples_per_label or getattr(
            self.metadata, "samples_per_label", 8
        )
        # kNN parameters
        self.k = k
        # Run metadata validation by instantiating addressing the attribute
        # This is quite hacky. Ideally, this would be done in the constructor of
        # each concrete task, but then we have to duplicate the __init__ method's
        # interface.
        if hasattr(self, "metadata"):
            self.metadata

    def _add_main_score(self, scores):
        if self.metadata.main_score in scores:
            scores["main_score"] = scores[self.metadata.main_score]
        else:
            logger.warn(
                f"main score {self.metadata.main_score} not found in scores {scores.keys()}"
            )

    def evaluate(self, model, eval_split="test", train_split="train", **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            scores = {}
            for lang in self.dataset:
                logger.info(
                    f"\nTask: {self.metadata.name}, split: {eval_split}, language: {lang}. Running..."
                )
                scores[lang] = self._evaluate_monolingual(
                    model, self.dataset[lang], eval_split, train_split, **kwargs
                )
                self._add_main_score(scores[lang])
        else:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {eval_split}. Running..."
            )
            scores = self._evaluate_monolingual(
                model, self.dataset, eval_split, train_split, **kwargs
            )
            self._add_main_score(scores)

        return scores

    def _evaluate_monolingual(
        self, model, dataset, eval_split="test", train_split="train", **kwargs
    ):
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {"k": self.k, "batch_size": self.batch_size}
        params.update(kwargs)

        scores = []
        test_cache, idxs = (
            None,
            None,
        )  # we store idxs to make the shuffling reproducible
        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split["label"], self.samples_per_label, idxs
            )
            train_samples.append(sample_indices)
        # Encode all unique sentences at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_sentences = train_split.select(unique_train_indices)["text"]
        unique_train_embeddings = dict(
            zip(unique_train_indices, model.encode(unique_train_sentences))
        )
        test_embeddings = model.encode(eval_split["text"])
        for i_experiment, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10
                + f" Experiment {i_experiment+1}/{self.n_experiments} "
                + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)["label"]
            binarizer = MultiLabelBinarizer()
            y_train = binarizer.fit_transform(y_train)
            y_test = binarizer.transform(eval_split["label"])
            scores_exp = evaluate_mlp(
                X_train,
                y_train,
                test_embeddings,
                y_test,
            )
            scores.append(scores_exp)

        if self.n_experiments == 1:
            return scores[0]
        else:
            avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
            std_errors = {
                k + "_stderr": np.std([s[k] for s in scores]) for k in scores[0].keys()
            }
            return {**avg_scores, **std_errors}

    def _undersample_data_indices(self, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        sample_indices = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if any((label_counter[label] < samples_per_label) for label in y[i]):
                sample_indices.append(i)
                for label in y[i]:
                    label_counter[label] += 1
        return sample_indices, idxs
