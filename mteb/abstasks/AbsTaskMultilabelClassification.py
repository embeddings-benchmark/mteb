from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ..MTEBResults import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


def evaluate_classifier(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin,
):
    scores = {}
    classifier = clone(classifier)
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

    classifier = KNeighborsClassifier(n_neighbors=5)

    def __init__(
        self,
        n_experiments=None,
        samples_per_label=None,
        batch_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        # Bootstrap parameters
        self.n_experiments = n_experiments or getattr(self, "n_experiments", 10)
        self.samples_per_label = samples_per_label or getattr(
            self, "samples_per_label", 8
        )
        # Run metadata validation by instantiating addressing the attribute
        # This is quite hacky. Ideally, this would be done in the constructor of
        # each concrete task, but then we have to duplicate the __init__ method's
        # interface.
        if hasattr(self, "metadata"):
            self.metadata

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self, model, dataset, eval_split="test", train_split="train", **kwargs
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {
            "classifier_type": type(self.classifier).__name__,
            "classifier_params": self.classifier.get_params(),
            "batch_size": self.batch_size,
        }
        params.update(kwargs)

        scores = []
        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split["label"], self.samples_per_label, None
            )
            train_samples.append(sample_indices)
        # Encode all unique sentences at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_sentences = train_split.select(unique_train_indices)["text"]
        unique_train_embeddings = dict(
            zip(unique_train_indices, model.encode(unique_train_sentences))
        )
        test_text = eval_split["text"]
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(eval_split["label"])
        # Stratified subsampling of test set to 2000 examples.
        try:
            if len(test_text) > 2000:
                test_text, _, y_test, _ = train_test_split(
                    test_text, y_test, stratify=y_test, train_size=2000
                )
        except ValueError:
            logger.warn("Couldn't subsample, continuing with the entire test set.")
        X_test = model.encode(test_text)
        for i_experiment, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10
                + f" Experiment {i_experiment+1}/{self.n_experiments} "
                + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)["label"]
            y_train = binarizer.transform(y_train)
            scores_exp = evaluate_classifier(
                X_train, y_train, X_test, y_test, self.classifier
            )
            scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores

        return avg_scores

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
