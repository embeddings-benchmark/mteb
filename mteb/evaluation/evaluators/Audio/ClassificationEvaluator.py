from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from torch.utils.data import DataLoader

from ..dataset_utils import AudioDataset, CustomAudioCollate
from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


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
        model_sampling_rate: int | None = None,  # Added to get sampling rate earlier
        model_max_audio_length_s: float
        | None = None,  # Added to get max length earlier
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        self.model_sampling_rate = (
            model_sampling_rate if model_sampling_rate is not None else 16000
        )
        self.model_max_audio_length_s = (
            model_max_audio_length_s if model_max_audio_length_s is not None else 30.0
        )

        self.dataset_train = AudioDataset(
            hf_dataset=dataset_train,
            audio_column_name=audio_column_name,
            target_sampling_rate=self.model_sampling_rate,
            mono=True,
        )

        self.y_train = dataset_train[label_column_name]
        self.dataset_test = AudioDataset(
            hf_dataset=dataset_test,
            audio_column_name=audio_column_name,
            target_sampling_rate=self.model_sampling_rate,
            mono=True,
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

        # Get model-specific parameters for collate_fn - now from self
        # model_sampling_rate = getattr(model, "sampling_rate", 16000)  # Default if not explicitly set
        # model_max_audio_length_s = getattr(model, "max_audio_length_s", 30.0) # Default if not explicitly set
        max_length_samples_for_collate = int(
            self.model_max_audio_length_s * self.model_sampling_rate
        )

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate, pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        # X_train = model.get_audio_embeddings(
        #     dataloader_train, batch_size=self.encode_kwargs["batch_size"]
        # )
        X_train_list = []
        for batch_data in dataloader_train:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                **self.encode_kwargs,
            )
            X_train_list.append(batch_embeddings)
        X_train = np.concatenate(X_train_list, axis=0)

        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate, pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        # X_test = model.get_audio_embeddings(
        #     dataloader, batch_size=self.encode_kwargs["batch_size"]
        # )
        X_test_list = []
        for batch_data in dataloader:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                **self.encode_kwargs,
            )
            X_test_list.append(batch_embeddings)
        X_test = np.concatenate(X_test_list, axis=0)
        test_cache = X_test

        logger.info("Fitting logistic regression classifier...")
        if X_train.dtype == np.dtype("float16") or X_train.dtype == np.dtype(
            "bfloat16"
        ):
            X_train = X_train.astype(np.float32)
        if X_test.dtype == np.dtype("float16") or X_test.dtype == np.dtype("bfloat16"):
            X_test = X_test.astype(np.float32)
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
