from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import (
    create_dataloader_from_texts,
    create_image_dataloader,
)
from mteb.models import Encoder
from mteb.similarity_functions import similarity

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class ZeroShotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        input_column_name: str,
        label_column_name: str,
        candidate_labels: list[str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset = dataset
        self.input_column_name = input_column_name
        self.labels = dataset[label_column_name]
        self.candidate_labels = candidate_labels
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self, model: Encoder, *, encode_kwargs: dict[str, Any]
    ) -> dict[str, float]:
        if "image" in self.task_metadata.modalities:
            dataloader = create_image_dataloader(
                self.dataset,
                image_column_name=self.input_column_name,
                batch_size=encode_kwargs["batch_size"],
            )
        elif self.task_metadata.modalities == ["text"]:
            dataloader = DataLoader(
                self.dataset, batch_size=encode_kwargs["batch_size"]
            )
        else:
            # To update for audio.
            raise ValueError(
                "ZeroShotClassificationEvaluator only supports image and text modalities."
            )

        text_label_embeddings = model.encode(
            create_dataloader_from_texts(self.candidate_labels),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        input_embeddings = model.encode(
            dataloader,
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        if self.task_metadata.modalities == ["text"]:
            probs = model.similarity(text_label_embeddings, input_embeddings)
        else:
            probs = similarity(text_label_embeddings, input_embeddings)

        predictions = probs.argmax(dim=1)
        accuracy = metrics.accuracy_score(self.labels, predictions.tolist())
        return {"accuracy": accuracy}
