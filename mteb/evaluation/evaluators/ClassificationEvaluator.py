from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.model_meta import ScoringFunction

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


class kNNClassificationEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        k: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

        self.k = k

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any] = {},
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = model.encode(
            DataLoader(self.train_dataset),
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        if test_cache is None:
            X_test = model.encode(
                DataLoader(self.eval_dataset),
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                **encode_kwargs,
            )
            test_cache = X_test
        else:
            X_test = test_cache

        y_train = self.train_dataset["label"]
        y_test = self.eval_dataset["label"]
        for metric in [
            ScoringFunction.COSINE,
            ScoringFunction.EUCLIDEAN,
        ]:  # TODO: "dot"
            knn = KNeighborsClassifier(
                n_neighbors=self.k, n_jobs=-1, metric=metric.value
            )
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            scores["accuracy_" + metric.value] = accuracy
            scores["f1_" + metric.value] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)  # type: ignore
            # if binary classification
            if len(np.unique(y_train)) == 2:
                ap = average_precision_score(y_test, y_pred)
                scores["ap_" + metric.value] = ap
                max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(y_train)) == 2:
            scores["ap"] = max_ap
        return scores, test_cache


class logRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        max_iter: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.max_iter = max_iter
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any] = {},
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        X_train = model.encode(
            DataLoader(self.train_dataset),
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        if test_cache is None:
            test_cache = model.encode(
                DataLoader(self.eval_dataset),
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                **encode_kwargs,
            )
        logger.info("Fitting logistic regression classifier...")
        y_train = self.train_dataset["label"]
        y_test = self.eval_dataset["label"]
        clf.fit(X_train, y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(test_cache)
        scores["accuracy"] = accuracy_score(y_test, y_pred)
        scores["f1"] = f1_score(y_test, y_pred, average="macro")
        scores["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")

        # if binary classification
        if len(np.unique(y_test)) == 2:
            scores["ap"] = average_precision_score(y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(
                y_test, y_pred, average="weighted"
            )

        return scores, test_cache
