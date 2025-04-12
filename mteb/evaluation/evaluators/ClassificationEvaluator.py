from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.model_meta import ScoringFunction

from ...create_dataloaders import create_image_dataloader
from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


class AbsClassificationEvaluator(Evaluator, ABC):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        values_column_name: str,
        label_column_name: str,
        is_image: bool,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        max_iter: int = 100,
        k: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.values_column_name = values_column_name
        self.label_column_name = label_column_name

        self.max_iter = max_iter
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.is_image = is_image
        self.k = k

    def create_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader]:
        if self.is_image:
            dataloader_train = create_image_dataloader(
                self.train_dataset,
                image_column_name=self.values_column_name,
                batch_size=batch_size,
            )
            dataloader_test = create_image_dataloader(
                self.eval_dataset,
                image_column_name=self.values_column_name,
                batch_size=batch_size,
            )
        else:
            if self.values_column_name != "text":
                self.train_dataset = self.train_dataset.rename_column(
                    self.values_column_name, "text"
                )
                self.eval_dataset = self.eval_dataset.rename_column(
                    self.values_column_name, "text"
                )
            dataloader_train = DataLoader(self.train_dataset)
            dataloader_test = DataLoader(self.eval_dataset)
        return dataloader_train, dataloader_test

    def calculate_scores(
        self,
        y_test: np.ndarray | list[int],
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        # if binary classification
        if len(np.unique(y_test)) == 2:
            scores["ap"] = average_precision_score(y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(
                y_test, y_pred, average="weighted"
            )
        return scores

    @abstractmethod
    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
        raise NotImplementedError()


class kNNClassificationEvaluator(AbsClassificationEvaluator):
    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
        scores = {}
        max_scores = {}
        dataloader_train, dataloader_test = self.create_dataloaders(
            batch_size=encode_kwargs["batch_size"]
        )
        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        if test_cache is None:
            X_test = model.encode(
                dataloader_test,
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
            ScoringFunction.COSINE.value,
            ScoringFunction.EUCLIDEAN.value,
            "l1",
        ]:
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            metric_scores = self.calculate_scores(y_test, y_pred)
            for metric_name, metric_score in metric_scores.items():
                scores[f"{metric_name}_{metric}"] = metric_score
                max_scores[metric_name] = max(
                    max_scores.get(metric_name, 0), metric_score
                )

        for metric_name, metric_score in max_scores.items():
            scores[f"{metric_name}_max"] = metric_score

        return scores, test_cache


class logRegClassificationEvaluator(AbsClassificationEvaluator):
    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )

        dataloader_train, dataloader_test = self.create_dataloaders(
            batch_size=encode_kwargs["batch_size"]
        )

        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        if test_cache is None:
            test_cache = model.encode(
                dataloader_test,
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                **encode_kwargs,
            )
        logger.info("Fitting logistic regression classifier...")
        y_train = self.train_dataset[self.label_column_name]
        y_test = self.eval_dataset[self.label_column_name]
        clf.fit(X_train, y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(test_cache)
        scores = self.calculate_scores(y_test, y_pred)
        return scores, test_cache
