from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from sklearn import metrics
from torch.utils.data import DataLoader
import torchaudio
import io


from mteb.encoder_interface import Encoder

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)

# transform = transforms.Compose([transforms.PILToTensor()])
# Replace with appropriate audio thing


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, audio_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.audio_column_name = audio_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx][self.audio_column_name]
        if isinstance(audio, bytes):
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio))
        elif isinstance(audio, str):
            # Assuming audio is a file path
            waveform, sample_rate = torchaudio.load(audio)
        else:
            # Assume audio is already a tensor or in a usable format
            waveform = audio
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform


def custom_collate_fn(batch):
    return batch


class ZeroshotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        audio_column_name: str,
        labels: list[int],  # Ground truth label indices
        candidate_labels: list[str],  # Text descriptions for zero-shot classes
        task_name: str | None = None,
        transform=None,
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize zero-shot audio classification evaluator.
        
        Args:
            dataset: HuggingFace dataset containing audio data
            audio_column_name: Name of column containing audio data
            labels: Ground truth labels (as indices into candidate_labels)
            candidate_labels: List of text descriptions for possible classes
            task_name: Optional name of the task
            transform: Optional audio transforms
            batch_size: Batch size for processing
        """
        super().__init__( **kwargs)
        self.dataset = AudioDataset(
            dataset, 
            audio_column_name=audio_column_name, 
            transform=transform
        )
        self.labels = labels
        self.candidate_labels = candidate_labels
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}) -> dict[str, float]:
        """Evaluate zero-shot classification performance.
        
        Args:
            model: An encoder model (like ClapZeroShotWrapper)
            encode_kwargs: Additional encoding arguments
            
        Returns:
            Dictionary containing accuracy metric
        """
        logger.info("Getting text embeddings for candidate labels...")
        text_embeddings = model.get_text_embeddings(self.candidate_labels)
        
        logger.info("Processing audio data...")
        dataloader = DataLoader(
            self.dataset,
            batch_size=encode_kwargs.get("batch_size", self.batch_size),
            collate_fn=custom_collate_fn
        )
        
        # Get audio embeddings
        audio_embeddings = model.get_audio_embeddings(dataloader)
        
        # Calculate similarity scores
        similarity = torch.from_numpy(audio_embeddings) @ torch.from_numpy(text_embeddings).T
        
        # Get predictions
        predictions = similarity.argmax(dim=1).cpu().numpy()
        
        # Calculate accuracy
        accuracy = metrics.accuracy_score(self.labels, predictions)
        
        return {"accuracy": accuracy}


class ZeroshotClassificationEvaluator(Evaluator):
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
        """
        super().__init__(**kwargs)
        self.dataset = AudioDataset(
            dataset, audio_column_name=audio_column_name, transform=transform
        )
        self.labels = dataset[label_column_name]
        self.candidate_labels = candidate_labels
        self.task_name = task_name
        self.batch_size = batch_size

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}) -> dict[str, float]:
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
        
        # Get audio embeddings
        audio_embeddings = model.get_audio_embeddings(dataloader)
        
        # Calculate similarity scores
        similarity = torch.from_numpy(audio_embeddings) @ torch.from_numpy(text_embeddings).T
        
        # Get predictions
        predictions = similarity.argmax(dim=1).cpu().numpy()
        
        # Calculate metrics
        scores = {
            "accuracy": metrics.accuracy_score(self.labels, predictions),
            "f1_macro": metrics.f1_score(self.labels, predictions, average="macro"),
            "f1_weighted": metrics.f1_score(self.labels, predictions, average="weighted")
        }
        
        
            
        return scores