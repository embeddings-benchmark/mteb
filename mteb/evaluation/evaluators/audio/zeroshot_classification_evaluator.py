import logging
from typing import Any

import torch
from sklearn import metrics

from mteb._create_dataloaders import (
    _create_audio_dataloader_from_audio_list,
    _create_dataloader_from_texts,
)
from mteb._evaluators import Evaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.models_protocols import EncoderProtocol

logger = logging.getLogger(__name__)


class AudioZeroshotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        audio_column_name: str,
        label_column_name: str,
        candidate_labels: list[str],
        task_metadata: TaskMetadata,
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
            task_metadata: Optional name of the task
            transform: Optional audio transforms
            batch_size: Batch size for processing
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.labels = dataset[label_column_name]
        self.candidate_labels = candidate_labels
        self.task_metadata = task_metadata
        self.batch_size = batch_size

    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any] = {}
    ) -> dict[str, float]:
        """Evaluate zero-shot classification performance."""
        logger.info("Getting text embeddings for candidate labels...")

        text_embeddings = model.encode(
            _create_dataloader_from_texts(self.candidate_labels),
            task_metadata=self.task_metadata,
            hf_subset="zeroshot_texts",
            hf_split="test",
            **encode_kwargs,
        )

        logger.info("Processing audio data...")
        dataloader = _create_audio_dataloader_from_audio_list(
            self.dataset["audio"], batch_size=self.batch_size
        )

        audio_embeddings = model.encode(
            dataloader,
            task_metadata=self.task_metadata,
            hf_subset="zeroshot_texts",
            hf_split="test",
            **encode_kwargs,
        )

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
