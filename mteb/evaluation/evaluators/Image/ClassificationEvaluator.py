from __future__ import annotations

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
from sklearn.neighbors import KNeighborsClassifier

from mteb.abstasks import TaskMetadata
from mteb.create_dataloaders import create_image_dataloader
from mteb.encoder_interface import Encoder
from mteb.model_meta import ScoringFunction
from mteb.similarity_functions import cos_sim, dot_score, euclidean_sim

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ImagekNNClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        image_column_name,
        label_column_name,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        k: int = 1,
        *,
        encode_kwargs: dict[str, Any],
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        self.image_column_name = image_column_name
        self.dataset_train = dataset_train
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = dataset_test
        self.y_test = dataset_test[label_column_name]
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.encode_kwargs = encode_kwargs

        self.k = k

    def __call__(self, model, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        dataloader_train = create_image_dataloader(
            self.dataset_train,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            batch_size=self.encode_kwargs["batch_size"],
        )
        dataloader = create_image_dataloader(
            self.dataset_test,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        if test_cache is None:
            X_test = model.encode(
                dataloader,
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                batch_size=self.encode_kwargs["batch_size"],
            )
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in [
            ScoringFunction.COSINE,
            ScoringFunction.EUCLIDEAN,
        ]:  # TODO: "dot"
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)  # type: ignore
            # if binary classification
            if len(np.unique(self.y_train)) == 2:
                ap = average_precision_score(self.y_test, y_pred)
                scores["ap_" + metric] = ap
                max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = max_ap
        return scores, test_cache


class ImagekNNClassificationEvaluatorPytorch(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        image_column_name,
        label_column_name,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        k: int = 1,
        *,
        encode_kwargs: dict[str, Any],
        limit: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        self.image_column_name = image_column_name
        self.dataset_train = dataset_train
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = dataset_test
        self.y_test = dataset_test[label_column_name]
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.encode_kwargs = encode_kwargs

        self.k = k

    def __call__(self, model: Encoder, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0

        dataloader_train = create_image_dataloader(
            self.dataset_train,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            batch_size=self.encode_kwargs["batch_size"],
        )

        dataloader = create_image_dataloader(
            self.dataset_test,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        if test_cache is None:
            X_test = model.encode(
                dataloader,
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                batch_size=self.encode_kwargs["batch_size"],
            )
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in [
            ScoringFunction.COSINE,
            ScoringFunction.EUCLIDEAN,
            ScoringFunction.DOT_PRODUCT,
            ScoringFunction.CUSTOM,
        ]:
            if metric == ScoringFunction.COSINE:
                distances = 1 - cos_sim(X_test, X_train)
            elif metric == ScoringFunction.EUCLIDEAN:
                distances = euclidean_sim(X_test, X_train)
            elif metric == ScoringFunction.DOT_PRODUCT:
                distances = -dot_score(X_test, X_train)
            elif metric == ScoringFunction.CUSTOM:
                distances = model.similarity(X_test, X_train)
            neigh_indices = torch.topk(
                distances, k=self.k, dim=1, largest=False
            ).indices
            y_train = torch.tensor(self.y_train)
            y_pred = torch.mode(
                y_train[neigh_indices], dim=1
            ).values  # TODO: case where there is no majority
            y_pred = y_pred.tolist()
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            scores["accuracy_" + metric.value] = accuracy
            scores["f1_" + metric.value] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)  # type: ignore
            # if binary classification
            if len(np.unique(self.y_train)) == 2:
                ap = average_precision_score(self.y_test, y_pred)
                scores["ap_" + metric.value] = ap
                max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = max_ap
        return scores, test_cache


class ImagelogRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        image_column_name,
        label_column_name,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        max_iter: int = 100,
        *,
        encode_kwargs: dict[str, Any],
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encode_kwargs = encode_kwargs

        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        self.image_column_name = image_column_name
        self.dataset_train = dataset_train
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = dataset_test
        self.y_test = dataset_test[label_column_name]

        self.max_iter = max_iter
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(self, model, test_cache=None):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )

        dataloader_train = create_image_dataloader(
            self.dataset_train,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        X_train = model.encode(
            dataloader_train,
            task_metadata=self.task_metadata,
            hf_split="train",
            hf_subset=self.hf_subset,
            batch_size=self.encode_kwargs["batch_size"],
        )

        dataloader = create_image_dataloader(
            self.dataset_test,
            image_column_name=self.image_column_name,
            batch_size=self.encode_kwargs["batch_size"],
        )
        if test_cache is None:
            X_test = model.encode(
                dataloader,
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                batch_size=self.encode_kwargs["batch_size"],
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
