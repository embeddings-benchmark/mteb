from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from torch.utils.data import DataLoader
from torchaudio import transforms

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


def get_resample_transform(original_sample_rate: int, target_sample_rate: int):
    return transforms.Resample(original_sample_rate, target_sample_rate)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        audio_column_name: str = "audio",
        transform: torch.nn.Module | None = None,  # anything from torchaudio.transforms
    ) -> None:
        self.dataset = hf_dataset
        self.transform = transform
        self.audio_column_name = audio_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio = self.dataset[idx][self.audio_column_name]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform


def custom_collate_fn(batch):
    return batch


class AudiologRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        audio_column_name,
        label_column_name,
        task_name: str,
        max_iter: int = 100,
        encode_kwargs: dict[str, Any] = {},
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        self.dataset_train = AudioDataset(
            dataset_train, audio_column_name=audio_column_name, transform=None
        )
        self.y_train = dataset_train[label_column_name]
        self.dataset_test = AudioDataset(
            dataset_test, audio_column_name=audio_column_name, transform=None
        )
        self.y_test = dataset_test[label_column_name]

        self.max_iter = max_iter
        self.task_name = task_name

    def __call__(self, model, test_cache=None):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        X_train = model.get_audio_embeddings(
            dataloader_train, batch_size=self.encode_kwargs["batch_size"]
        )
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        if test_cache is None:
            X_test = model.get_audio_embeddings(
                dataloader, batch_size=self.encode_kwargs["batch_size"]
            )
            test_cache = X_test
        else:
            X_test = test_cache
        logger.info("Fitting logistic regression classifier...")
        if X_train.dtype == torch.bfloat16:
            X_train = X_train.to(torch.float32)
        if X_test.dtype == torch.bfloat16:
            X_test = X_test.to(torch.float32)
        clf.fit(X_train, self.y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        scores["accuracy"] = accuracy_score(self.y_test, y_pred)
        scores["f1"] = f1_score(self.y_test, y_pred, average="macro")
        scores["f1_weighted"] = f1_score(self.y_test, y_pred, average="weighted")

        # if binary classification
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = average_precision_score(self.y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(
                self.y_test, y_pred, average="weighted"
            )

        return scores, test_cache
