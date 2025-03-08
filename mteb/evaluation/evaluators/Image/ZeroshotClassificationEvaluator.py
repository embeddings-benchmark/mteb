from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from datasets import Dataset
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms

from mteb.create_dataloaders import create_dataloader_from_texts
from mteb.encoder_interface import Encoder

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)

def convert_images_to_rgb(example: dict[str, Any]) -> dict[str, Any]:
    image = example["image"]
    # For PIL images
    if hasattr(image, "mode") and image.mode != "RGB":
        example["image"] = image.convert("RGB")
    # For tensor images with 1 channel
    elif isinstance(image, torch.Tensor) and image.shape[0] == 1:
        example["image"] = image.repeat(3, 1, 1)
    return example


class ZeroshotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        image_column_name: str,
        labels: list[int],
        candidate_labels: list[str],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "image" not in dataset.column_names:
            dataset = dataset.rename_column(image_column_name, "image")
        self.dataset = dataset.map(
            convert_images_to_rgb,
            desc="Converting images to RGB"
        ).with_format("torch")
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
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        text_embeddings = model.encode(
            create_dataloader_from_texts(self.candidate_labels),
            task_name=self.task_name,
            batch_size=encode_kwargs["batch_size"],
        )

        image_embeddings = model.encode(
            dataloader,
            task_name=self.task_name,
            batch_size=encode_kwargs["batch_size"],
        )

        # todo change to similarity
        probs = model.calculate_probs(text_embeddings, image_embeddings)
        predictions = probs.argmax(dim=1)

        logger.info("Evaluating...")

        accuracy = metrics.accuracy_score(self.labels, predictions.tolist())

        return {"accuracy": accuracy}
