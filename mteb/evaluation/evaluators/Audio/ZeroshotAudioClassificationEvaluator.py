from __future__ import annotations

import io
import logging
import math
import os
from typing import Any

import torch
import torchaudio
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)

def id(x):
    return x

DEFAULT_AUDIO_TRANSFORM = id


# pasted audio dataset into this
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, audio_column_name: str = "audio", sampling_rate: int = 48000, transform=None):
        self.dataset = hf_dataset
        self.transform = transform or DEFAULT_AUDIO_TRANSFORM
        self.audio_column_name = audio_column_name
        self.sampling_rate = sampling_rate

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


class ZeroshotAudioClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        audio_column_name: str,
        labels: list[int],
        candidate_labels: list[str],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = AudioDataset(
            dataset, audio_column_name=audio_column_name, transform=None
        )
        self.audio_column_name = audio_column_name
        self.labels = labels
        self.candidate_labels = candidate_labels
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        text_embeddings = model.get_text_embeddings(
            self.candidate_labels, batch_size=encode_kwargs["batch_size"]
        )

        audio_embeddings = model.get_audio_embeddings(
            self.dataset, sampling_rate=self.dataset.sampling_rate, batch_size=encode_kwargs["batch_size"]
        )

        probs = model.calculate_probs(text_embeddings, audio_embeddings)
        predictions = probs.argmax(dim=1)

        logger.info("Evaluating...")

        accuracy = metrics.accuracy_score(self.labels, predictions.tolist())

        return {"accuracy": accuracy}
