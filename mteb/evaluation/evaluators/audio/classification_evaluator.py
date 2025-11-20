import logging
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)

from mteb._create_dataloaders import _create_audio_dataloader_from_audio_list
from mteb._evaluators import Evaluator
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


def get_resample_transform(original_sample_rate: int, target_sample_rate: int):
    from torchaudio import transforms

    return transforms.Resample(original_sample_rate, target_sample_rate)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


class AudiologRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        audio_column_name,
        label_column_name,
        task_metadata: TaskMetadata,
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

        self.dataset_train = dataset_train
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = dataset_test
        self.y_test = dataset_test[label_column_name]

        self.max_iter = max_iter
        self.task_metadata = task_metadata

    def __call__(self, model, test_cache=None):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        dataloader_train = _create_audio_dataloader_from_audio_list(
            self.dataset_train["audio"]
        )
        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset="default",
            batch_size=self.encode_kwargs["batch_size"],
        )
        dataloader = _create_audio_dataloader_from_audio_list(
            self.dataset_test["audio"]
        )
        X_test = model.encode(
            dataloader,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset="default",
            batch_size=self.encode_kwargs["batch_size"],
        )
        test_cache = X_test

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
