from .AbsTask import AbsTask
import datasets
from ..evaluation.evaluators import (
    kNNClassificationEvaluator,
    logRegClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
)
import numpy as np
import logging
from collections import defaultdict


class AbsTaskClassification(AbsTask):
    """
    Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for classification. #TODO:
    """

    def __init__(self, **kwargs):
        super(AbsTaskClassification, self).__init__(**kwargs)
        self.method = kwargs.get("method", "logReg")

        # Bootstrap parameters
        self.n_splits = kwargs.get("n_splits", self.description.get("n_splits", 1))
        self.samples_per_label = kwargs.get(
            "samples_per_label", self.description.get("samples_per_label", float("inf"))
        )

        # kNN parameters
        self.k = kwargs.get("k", 3)

    def evaluate(self, model, eval_split="test", train_split="train"):
        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            scores = {}
            for lang in self.langs:
                print(f"\nTask: {self.description['name']}, split: {eval_split}, language: {lang}. Running...")
                scores[lang] = self._evaluate_monolingual(model, self.dataset[lang], eval_split, train_split)
        else:
            print(f"\nTask: {self.description['name']}, split: {eval_split}. Running...")
            scores = self._evaluate_monolingual(model, self.dataset, eval_split, train_split)

        if self.description["main_score"] in scores:
            scores["main_score"] = scores[self.description["main_score"]]
        else:
            print(f"WARNING: main score {self.description['main_score']} not found in scores {scores.keys()}")

        return scores

    def _evaluate_monolingual(self, model, dataset, eval_split="test", train_split="train"):
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]

        logging.getLogger("sentence_transformers.evaluation.ClassificationEvaluator").setLevel(logging.WARN)

        scores = []
        idxs = None  # we store idxs to make the shuffling reproducible
        for _ in range(self.n_splits):
            # Bootstrap `self.samples_per_label` samples per label for each split
            X_sampled, y_sampled, idxs = self._undersample_data(
                train_split["text"], train_split["label"], self.samples_per_label, idxs
            )

            if self.method == "kNN":
                evaluator = kNNClassificationEvaluator(
                    X_sampled, y_sampled, eval_split["text"], eval_split["label"], k=self.k
                )
            elif self.method == "kNN-pytorch":
                evaluator = kNNClassificationEvaluatorPytorch(
                    X_sampled, y_sampled, eval_split["text"], eval_split["label"], k=self.k
                )
            elif self.method == "logReg":
                evaluator = logRegClassificationEvaluator(
                    X_sampled, y_sampled, eval_split["text"], eval_split["label"]
                )
            else:
                raise ValueError(f"Method {self.method} not supported")

            scores.append(evaluator(model))

        if self.n_splits == 1:
            return scores[0]
        else:
            avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
            std_errors = {k + "_stderr": np.std([s[k] for s in scores]) for k in scores[0].keys()}
            return {**avg_scores, **std_errors}

    def _undersample_data(self, X, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)  # TODO: fix reproducibility
        label_counter = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs
