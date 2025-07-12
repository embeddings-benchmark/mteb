from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder

from ..dataset_utils import AudioDataset, custom_collate_fn
from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


class AudioZeroshotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        audio_column_name: str,
        label_column_name: str,
        candidate_labels: list[str],
        task_name: str | None = None,
        transform=None,
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize zero-shot audio classification evaluator.

        Args:
            dataset: HuggingFace dataset containing audio data
            audio_column_name: Name of column containing audio data
            label_column_name: Name of column containing label indices
            candidate_labels: List of text descriptions for possible classes
            task_name: Optional name of the task
            transform: Optional audio transforms
            batch_size: Batch size for processing
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.dataset = AudioDataset(
            dataset, audio_column_name=audio_column_name, transform=transform
        )
        self.labels = dataset[label_column_name]
        self.candidate_labels = candidate_labels
        self.task_name = task_name
        self.batch_size = batch_size

    def __call__(
        self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}
    ) -> dict[str, float]:
        """Evaluate zero-shot classification performance."""
        logger.info("Getting text embeddings for candidate labels...")

        text_embeddings = model.get_text_embeddings(self.candidate_labels)

        logger.info("Processing audio data...")
        dataloader = DataLoader(
            self.dataset,
            batch_size=encode_kwargs.get("batch_size", self.batch_size),
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        audio_embeddings = model.get_audio_embeddings(dataloader)

        # Calculate similarity scores
        similarity = (
            torch.from_numpy(audio_embeddings) @ torch.from_numpy(text_embeddings).T
        )

        predictions = similarity.argmax(dim=1).cpu().numpy()

        # Calculate metrics
        scores = {
            "accuracy": metrics.accuracy_score(self.labels, predictions),
            "f1": metrics.f1_score(self.labels, predictions, average="macro"),
            "f1_weighted": metrics.f1_score(self.labels, predictions, average="macro"),
            "precision": metrics.precision_score(
                self.labels, predictions, average="macro"
            ),
            "recall": metrics.recall_score(self.labels, predictions, average="macro"),
        }

        return scores
