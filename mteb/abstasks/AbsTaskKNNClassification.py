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


class AbsTaskKNNClassification(AbsTask):
    """
    Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for classification. #TODO:
    """

    def __init__(self, **kwargs):
        super(AbsTaskKNNClassification, self).__init__(**kwargs)
        self.method = kwargs.get("method", "logReg-10-splits-5-intents")
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
            scores = self._evaluate_monolingual(model, self.dataset, eval_split, train_split)

        if self.description["main_score"] in scores:
            scores["main_score"] = scores[self.description["main_score"]]
        else:
            print(f"WARNING: main score {self.description['main_score']} not found in scores {scores.keys()}")

        return scores

    def _evaluate_monolingual(self, model, dataset, eval_split="test", train_split="train"):
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]

        logging.getLogger("sentence_transformers.evaluation.kNNClassificationEvaluator").setLevel(logging.WARN)
        if self.method == "kNN":
            evaluator = kNNClassificationEvaluator(
                train_split["text"], train_split["label"], eval_split["text"], eval_split["label"], k=self.k
            )
        elif self.method == "kNN-pytorch":
            evaluator = kNNClassificationEvaluatorPytorch(
                train_split["text"], train_split["label"], eval_split["text"], eval_split["label"], k=self.k
            )
        elif self.method == "logReg":
            evaluator = logRegClassificationEvaluator(
                train_split["text"], train_split["label"], eval_split["text"], eval_split["label"]
            )
        elif self.method == "logReg-10-splits-5-intents":
            n_splits = 10
            samples_per_label = 5

            # we only keep 5 samples for n_splits iterations
            avg_scores = defaultdict(float)
            idxs = None  # we store idxs to make the shuffling reproducible
            for _ in range(n_splits):
                X_sampled, y_sampled, idxs = self._undersample_data(
                    train_split["text"], train_split["label"], samples_per_label, idxs
                )
                evaluator = logRegClassificationEvaluator(X_sampled, y_sampled, eval_split["text"], eval_split["label"])
                scores = evaluator(model)
                avg_scores = {k: avg_scores[k] + scores[k] / n_splits for k in scores}

            return avg_scores

        else:
            raise ValueError(f"Method {self.method} not supported")
        scores = evaluator(model)
        return scores

    def _undersample_data(self, X, y, samples_per_label, idxs=None):
        """ Undersample data to have samples_per_label samples of each label """
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs
