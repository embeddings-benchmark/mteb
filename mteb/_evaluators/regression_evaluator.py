from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from datasets import Dataset
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from typing_extensions import Self

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import create_image_dataloader
from mteb.models import Encoder
from mteb.types import BatchedInput

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class SklearnRegressorModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None: ...  # noqa: N803
    def predict(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803
    def get_params(self) -> dict[str, Any]: ...
    def set_params(self, **kwargs: dict[str, Any]) -> Self: ...


class LinearRegressionEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        values_column_name: str,
        label_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        regressor: SklearnRegressorModel,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.values_column = values_column_name
        self.label_column_name = label_column_name

        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.task_metadata = task_metadata
        self.regressor = regressor

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        y_train = self.train_dataset[self.label_column_name]
        y_test = self.eval_dataset[self.label_column_name]

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

        self.regressor.fit(X_train, y_train)
        y_pred = self.regressor.predict(test_cache)
        scores = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "kendalltau": kendalltau(y_test, y_pred).statistic,
        }
        scores["rmse"] = np.sqrt(scores["mse"])

        return scores, test_cache

    def create_dataloaders(
        self, batch_size: int
    ) -> tuple[DataLoader[BatchedInput], DataLoader[BatchedInput]]:
        if self.task_metadata.modalities == ["image"]:
            dataloader_train = create_image_dataloader(
                self.train_dataset,
                image_column_name=self.label_column_name,
                batch_size=batch_size,
            )
            dataloader_test = create_image_dataloader(
                self.eval_dataset,
                image_column_name=self.label_column_name,
                batch_size=batch_size,
            )
        elif self.task_metadata.modalities == ["text"]:
            if self.label_column_name != "text":
                self.train_dataset = self.train_dataset.rename_column(
                    self.label_column_name, "text"
                )
                self.eval_dataset = self.eval_dataset.rename_column(
                    self.label_column_name, "text"
                )
            dataloader_train = DataLoader(self.train_dataset)
            dataloader_test = DataLoader(self.eval_dataset)
        else:
            raise ValueError(
                "ClassificationEvaluator only supports image and text modalities."
            )
        return dataloader_train, dataloader_test
