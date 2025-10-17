import logging
from typing import Any, Protocol

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from typing_extensions import Self

from mteb._create_dataloaders import _create_image_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol
from mteb.types import BatchedInput

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class SklearnModelProtocol(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray | list[int]) -> None: ...  # noqa: N803
    def predict(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803
    def get_params(self) -> dict[str, Any]: ...
    def set_params(self, **kwargs: dict[str, Any]) -> Self: ...
    def score(self, X: np.ndarray, y: np.ndarray | list[int]) -> float: ...  # noqa: N803


class SklearnEvaluator(Evaluator):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        values_column_name: str,
        label_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        evaluator_model: SklearnModelProtocol,
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
        self.evaluator_model = evaluator_model

    def create_dataloaders(
        self, batch_size: int
    ) -> tuple[DataLoader[BatchedInput], DataLoader[BatchedInput]]:
        if self.task_metadata.modalities == ["image"]:
            dataloader_train = _create_image_dataloader(
                self.train_dataset,
                image_column_name=self.values_column_name,
                batch_size=batch_size,
            )
            dataloader_test = _create_image_dataloader(
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

    def __call__(  # type: ignore[override]
        self,
        model: EncoderProtocol,
        *,
        encode_kwargs: dict[str, Any],
        test_cache: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classification evaluation by training a sklearn classifier on the embeddings of the training set and evaluating on the embeddings of the test set.

        Args:
            model: Encoder
            encode_kwargs: encode kwargs
            test_cache: embeddings of the test set, if already computed

        Returns:
            Tuple of test predictions and embeddings

        """
        dataloader_train, dataloader_test = self.create_dataloaders(
            batch_size=encode_kwargs["batch_size"]
        )

        logger.info("Running - Encoding samples...")
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

        logger.info("Running - Fitting classifier...")
        y_train = self.train_dataset[self.label_column_name]
        self.evaluator_model.fit(X_train, y_train)

        logger.info("Running - Evaluating classifier...")
        y_pred = self.evaluator_model.predict(test_cache)
        return y_pred, test_cache
