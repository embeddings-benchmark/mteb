from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.create_dataloaders import (
    transform_image_to_rgb,
)
from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.Evaluator import Evaluator
from mteb.requires_package import requires_image_dependencies

logger = logging.getLogger(__name__)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: list[torch.Tensor],
    ):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
        }

    @property
    def features(self) -> dict[str, Any]:
        # for correct wrapper handling
        return {"image": []}


class ImageTextPairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of
    identifying similar and dissimilar image caption pairs.
    The goal is to find the correct image for each caption and the correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        images: Each row is a list of images.
        texts: Each row is a list of captions.
        batch_size: Batch size used to compute embeddings
    """

    def __init__(
        self,
        dataset,
        images_column_names: str | list[str],
        texts_column_names: str | list[str],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        requires_image_dependencies()
        from torchvision import transforms

        self.dataset = dataset
        self.images_column_names = images_column_names
        self.texts_column_names = texts_column_names
        self.task_name = task_name
        self.transform = transforms.Compose([transforms.PILToTensor()])

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 64

        num_images_per_sample = (
            len(self.images_column_names)
            if isinstance(self.images_column_names, list)
            else 1
        )
        num_texts_per_sample = (
            len(self.texts_column_names)
            if isinstance(self.texts_column_names, list)
            else 1
        )

        images = []
        if isinstance(self.images_column_names, list):
            for row in self.dataset:
                for col in self.images_column_names:
                    images.append(row[col])
        else:
            images = self.dataset[self.images_column_names]

        images = [self.transform(transform_image_to_rgb(img)) for img in images]

        texts = []
        if isinstance(self.texts_column_names, list):
            for row in self.dataset:
                for col in self.texts_column_names:
                    texts.append(row[col])
        else:
            texts = self.dataset[self.texts_column_names]

        img_ground_truths = torch.arange(num_images_per_sample)
        caption_ground_truths = torch.arange(num_texts_per_sample)

        text_embeddings = model.encode(
            DataLoader(
                Dataset.from_dict({"text": texts}),
                batch_size=encode_kwargs["batch_size"],
            ),
            task_name=self.task_name,
            **encode_kwargs,
        )

        norm_text_embeddings = F.normalize(
            text_embeddings,
            dim=-1,
        ).view(len(self.dataset), num_texts_per_sample, -1)

        image_embeddings = model.encode(
            DataLoader(
                CustomImageDataset(images),
                batch_size=encode_kwargs["batch_size"],
                collate_fn=lambda x: {"image": [item["image"] for item in x]},
            ),
            task_name=self.task_name,
            **encode_kwargs,
        )

        norm_image_embeddings = F.normalize(
            image_embeddings,
            dim=-1,
        ).view(len(self.dataset), num_images_per_sample, -1)

        image_score = []
        text_score = []
        score = []

        for img_emb, txt_emb in zip(norm_image_embeddings, norm_text_embeddings):
            scores = (
                img_emb @ txt_emb.t()
            )  # shape = (num_images_per_sample x num_texts_per_sample)

            image_closest_text = scores.argmax(dim=1)  # shape = (num_images_per_sample)
            text_closest_image = scores.argmax(dim=0)  # shape = (num_texts_per_sample)
            pred_text_is_correct = (
                (image_closest_text == img_ground_truths).all().item()
            )
            pred_image_is_correct = (
                (text_closest_image == caption_ground_truths).all().item()
            )
            all_correct = pred_text_is_correct and pred_image_is_correct
            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            score.append(all_correct)

        metrics = {}
        metrics["image_acc"] = torch.Tensor(image_score).float().mean().item()
        metrics["text_acc"] = torch.Tensor(text_score).float().mean().item()
        metrics["accuracy"] = torch.Tensor(score).float().mean().item()
        return metrics
