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
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder
from mteb.requires_package import requires_image_dependencies

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


def dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -np.dot(a, b)


def get_default_transform():
    requires_image_dependencies()
    from torchvision import transforms

    return transforms.Compose([transforms.PILToTensor()])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        return image


def custom_collate_fn(batch):
    return batch


class ImagekNNClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        image_column_name,
        label_column_name,
        task_name: str | None = None,
        k: int = 1,
        encode_kwargs: dict[str, Any] = {},
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        default_transform = get_default_transform()
        self.dataset_train = ImageDataset(
            dataset_train,
            image_column_name=image_column_name,
            transform=default_transform,
        )
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = ImageDataset(
            dataset_test,
            image_column_name=image_column_name,
            transform=default_transform,
        )
        self.y_test = dataset_test[label_column_name]
        self.task_name = task_name
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

        self.k = k

    def __call__(self, model, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        X_train = model.get_image_embeddings(
            dataloader_train, batch_size=self.encode_kwargs["batch_size"]
        )
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        if test_cache is None:
            X_test = model.get_image_embeddings(
                dataloader, batch_size=self.encode_kwargs["batch_size"]
            )
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean"]:  # TODO: "dot"
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
        task_name: str,
        k: int = 1,
        encode_kwargs: dict[str, Any] = {},
        limit: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            dataset_train = dataset_train.select(list(range(limit)))

        default_transform = get_default_transform()
        self.dataset_train = ImageDataset(
            dataset_train,
            image_column_name=image_column_name,
            transform=default_transform,
        )
        self.y_train = dataset_train[label_column_name]

        self.dataset_test = ImageDataset(
            dataset_test,
            image_column_name=image_column_name,
            transform=default_transform,
        )
        self.y_test = dataset_test[label_column_name]
        self.task_name = task_name
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

        self.k = k

    def __call__(self, model: Encoder, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        X_train = model.get_image_embeddings(
            dataloader_train, batch_size=self.encode_kwargs["batch_size"]
        )

        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        if test_cache is None:
            X_test = model.get_image_embeddings(
                dataloader, batch_size=self.encode_kwargs["batch_size"]
            )
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean", "dot"]:
            if metric == "cosine":
                distances = 1 - self._cos_sim(X_test, X_train)
            elif metric == "euclidean":
                distances = self._euclidean_dist(X_test, X_train)
            elif metric == "dot":
                distances = -self._dot_score(X_test, X_train)
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

    @staticmethod
    def _cos_sim(a: Tensor, b: Tensor):
        """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

        Return:
            Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euclidean_dist(a: Tensor, b: Tensor):
        """Computes the euclidean distance euclidean_dist(a[i], b[j]) for all i and j.

        Returns:
            Matrix with res[i][j]  = euclidean_dist(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.cdist(a, b, p=2)

    @staticmethod
    def _dot_score(a: Tensor, b: Tensor):
        """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

        Returns:
            Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))


class ImagelogRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset_train,
        dataset_test,
        image_column_name,
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

        default_transform = get_default_transform()
        self.dataset_train = ImageDataset(
            dataset_train,
            image_column_name=image_column_name,
            transform=default_transform,
        )
        self.y_train = dataset_train[label_column_name]
        self.dataset_test = ImageDataset(
            dataset_test,
            image_column_name=image_column_name,
            transform=default_transform,
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
        X_train = model.get_image_embeddings(
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
            X_test = model.get_image_embeddings(
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
