from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.encoder_interface import Encoder

from ...create_dataloaders import create_image_dataloader
from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ClassificationEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        values_column_name: str,
        label_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        classifier: BaseEstimator,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.values_column_name = values_column_name
        self.label_column_name = label_column_name

        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.classifier = classifier

    def create_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader]:
        if self.task_metadata.modalities == ["image"]:
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
        elif self.task_metadata.modalities == ["text"]:
            if self.values_column_name != "text":
                self.train_dataset = self.train_dataset.rename_column(
                    self.values_column_name, "text"
                )
                self.eval_dataset = self.eval_dataset.rename_column(
                    self.values_column_name, "text"
                )
            dataloader_train = DataLoader(self.train_dataset)
            dataloader_test = DataLoader(self.eval_dataset)
        else:
            raise ValueError(
                "ClassificationEvaluator only supports image and text modalities."
            )
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

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], Any]:
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
        self.classifier.fit(X_train, y_train)
        logger.info("Evaluating...")
        y_pred = self.classifier.predict(test_cache)
        scores = self.calculate_scores(y_test, y_pred)
        return scores, test_cache
