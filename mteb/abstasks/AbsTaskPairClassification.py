from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


from datasets import Dataset

from ..encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from ..MTEBResults import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


def _undersample_data(X, y, samples_per_label: int, idxs=None):
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


def _prepare_features(
    dataset, model, eval_split="test", train_split="train", samples_per_label: int = 8
):
    test_sent1 = dataset[eval_split]["sent1"]
    test_sent2 = dataset[eval_split]["sent1"]
    test_emb1 = model.encode(test_sent1)
    test_emb2 = model.encode(test_sent2)
    y_test = dataset[eval_split]["labels"]
    X_test = np.concatenate(
        (test_emb1, test_emb2, np.abs(test_emb1 - test_emb2)), axis=1
    )
    train_sent1 = dataset[train_split]["sent1"]
    train_sent2 = dataset[train_split]["sent1"]
    train_emb1 = model.encode(train_sent1)
    train_emb2 = model.encode(train_sent2)
    y_train = dataset[train_split]["labels"]
    X_train = np.concatenate(
        (train_emb1, train_emb2, np.abs(train_emb1 - train_emb2)), axis=1
    )
    X_train, y_train, _ = _undersample_data(X_train, y_train, samples_per_label)
    return X_train, X_test, y_train, y_test


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sent1: list[str]
        sent2: list[str]
        labels: list[int]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_subset(
        self, model: Encoder | EncoderWithQueryCorpusEncode, dataset: Dataset, eval_split="test", train_split="train", **kwargs
    ) -> ScoresDict:
        X_train, X_test, y_train, y_test = _prepare_features(
            dataset, model, eval_split, train_split
        )
        classifier = LogisticRegression().fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores = {
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "ap": average_precision_score(y_test, y_pred),
        }
        return scores

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()
        if self.is_multilingual:
            scores = dict()
            print("loaded langs:", self.dataset.keys())
            for lang, monolingual_dataset in self.dataset.items():
                logger.info(
                    f"\nTask: {self.metadata_dict['name']}, split: {split}, language: {lang}. Running..."
                )
                scores[lang] = self._evaluate_monolingual(
                    model, monolingual_dataset, split=split, **kwargs
                )
                self._add_main_score(scores[lang])
            return scores
        else:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}. Running..."
            )
            scores = self._evaluate_monolingual(
                model, self.dataset, split=split, **kwargs
            )
            self._add_main_score(scores)
            return scores

    def _add_main_score(self, scores):
        if self.metadata.main_score in scores:
            scores["main_score"] = scores[self.metadata.main_score]
        else:
            logger.warn(
                f"main score {self.metadata.main_score} not found in scores {scores.keys()}"
            )