from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder

from ..dataset_utils import AudioDataset, CustomAudioCollate
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
        model_sampling_rate: int | None = None, # Added to get sampling rate earlier
        model_max_audio_length_s: float | None = None, # Added to get max length earlier
        **kwargs,
    ):
        """Initialize zero-shot audio classification evaluator."""
        super().__init__(**kwargs)
        
        self.model_sampling_rate = model_sampling_rate if model_sampling_rate is not None else 16000
        self.model_max_audio_length_s = model_max_audio_length_s if model_max_audio_length_s is not None else 30.0

        self.dataset = AudioDataset(
            hf_dataset=dataset, 
            audio_column_name=audio_column_name, 
            target_sampling_rate=self.model_sampling_rate,
            mono=True,
            transform=transform # Keep any additional transforms
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

        # Get model-specific parameters for collate_fn - now from self
        # model_sampling_rate = getattr(model, "sampling_rate", 16000)  # Default if not explicitly set
        # model_max_audio_length_s = getattr(model, "max_audio_length_s", 30.0) # Default if not explicitly set
        max_length_samples_for_collate = int(self.model_max_audio_length_s * self.model_sampling_rate)

        dataloader = DataLoader(
            self.dataset,
            batch_size=encode_kwargs.get("batch_size", self.batch_size),
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate,
                pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        # audio_embeddings = model.get_audio_embeddings(dataloader)
        audio_embeddings_list = []
        for batch_data in dataloader:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                **encode_kwargs,
            )
            audio_embeddings_list.append(batch_embeddings)
        audio_embeddings = torch.cat(audio_embeddings_list, dim=0).cpu().numpy()

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
