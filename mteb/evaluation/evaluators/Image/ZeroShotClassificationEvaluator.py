from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder
from mteb.requires_package import requires_image_dependencies

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


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


class ZeroShotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        image_column_name: str,
        labels: list[int],
        candidate_labels: list[str],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = ImageDataset(
            dataset,
            image_column_name=image_column_name,
            transform=get_default_transform(),
        )
        self.image_column_name = image_column_name
        self.labels = labels
        self.candidate_labels = candidate_labels
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        dataloader = DataLoader(
            self.dataset,
            batch_size=encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        text_embeddings = model.get_text_embeddings(
            self.candidate_labels, batch_size=encode_kwargs["batch_size"]
        )

        image_embeddings = model.get_image_embeddings(
            dataloader, batch_size=encode_kwargs["batch_size"]
        )

        probs = model.calculate_probs(text_embeddings, image_embeddings)
        predictions = probs.argmax(dim=1)

        logger.info("Evaluating...")

        accuracy = metrics.accuracy_score(self.labels, predictions.tolist())

        return {"accuracy": accuracy}
