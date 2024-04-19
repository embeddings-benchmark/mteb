from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import MultilabelBinarizer

from ..evaluation.evaluators import kNNMultiLabelClassificationEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskMultilabelClassification(AbsTask):
    """
    Abstract class for kNN multioutput classification tasks
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
        self.n_experiments = (
            n_experiments
            if n_experiments is not None
            else self.metadata_dict.get("n_experiments", 10)
        )
        self.samples_per_label = (
            samples_per_label
            if samples_per_label is not None
            else self.metadata_dict.get("samples_per_label", 8)
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
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warn(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )

    def evaluate(self, model, eval_split="test", train_split="train", **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            scores = {}
            for lang in self.dataset:
                logger.info(
                    f"\nTask: {self.metadata_dict['name']}, split: {eval_split}, language: {lang}. Running..."
                )
                scores[lang] = self._evaluate_monolingual(
                    model, self.dataset[lang], eval_split, train_split, **kwargs
                )
                self._add_main_score(scores[lang])
        else:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {eval_split}. Running..."
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
        for i in range(self.n_experiments):
            logger.info(
                "=" * 10 + f" Experiment {i+1}/{self.n_experiments} " + "=" * 10
            )
            # Bootstrap `self.samples_per_label` samples per label for each split
            X_sampled, y_sampled, idxs = self._undersample_data(
                train_split["text"], train_split["label"], self.samples_per_label, idxs
            )
            binarizer = MultilabelBinarizer()
            y_train = binarizer.fit_transform(y_sampled)
            y_test = binarizer.transform(eval_split["label"])
            evaluator = kNNMultiLabelClassificationEvaluator(
                X_sampled,
                y_train,
                eval_split["text"],
                y_test,
                **params,
            )
            scores_exp, test_cache = evaluator(model, test_cache=test_cache)
            scores.append(scores_exp)

        if self.n_experiments == 1:
            return scores[0]
        else:
            avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
            std_errors = {
                k + "_stderr": np.std([s[k] for s in scores]) for k in scores[0].keys()
            }
            return {**avg_scores, **std_errors}

    def _undersample_data(self, X, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if any(label_counter[label] for label in y[i]) < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                for label in y[i]:
                    label_counter[label] += 1
        return X_sampled, y_sampled, idxs
