from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, images_column_names, texts_column_names, transform=None
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.images_column_names = images_column_names
        self.texts_column_names = texts_column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Get images
        if isinstance(self.images_column_names, str):
            images = data[self.images_column_names]
        else:
            images = [data[col] for col in self.images_column_names]

        # Apply transforms to images
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        # Get texts
        if isinstance(self.texts_column_names, str):
            texts = data[self.texts_column_names]
        else:
            texts = [data[col] for col in self.texts_column_names]

        return images, texts


def custom_collate_fn(batch):
    return batch


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
        transform=None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            dataset = dataset.select(range(limit))
        self.dataset = dataset
        self.images_column_names = images_column_names
        self.texts_column_names = texts_column_names
        self.task_name = task_name
        self.transform = transform

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 64

        data_loader = DataLoader(
            ImageTextDataset(
                self.dataset,
                self.images_column_names,
                self.texts_column_names,
                transform=self.transform,
            ),
            batch_size=encode_kwargs["batch_size"],
            shuffle=False,
            # collate_fn=lambda x: x,  # Identity collate function
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

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

        img_ground_truths = torch.arange(num_images_per_sample)
        caption_ground_truths = torch.arange(num_texts_per_sample)

        image_score = []
        text_score = []
        score = []

        for batch in data_loader:
            images_list, texts_list = zip(*batch)
            images = [img for images in images_list for img in images]
            texts = [txt for texts in texts_list for txt in texts]
            images_emb = F.normalize(
                model.get_image_embeddings(
                    images, batch_size=len(images), task_name=self.task_name
                ),
                dim=-1,
            ).view(len(batch), num_images_per_sample, -1)
            texts_emb = F.normalize(
                model.get_text_embeddings(
                    texts, batch_size=len(texts), task_name=self.task_name
                ),
                dim=-1,
            ).view(len(batch), num_texts_per_sample, -1)
            for i in range(len(batch)):
                img_emb = images_emb[i]
                txt_emb = texts_emb[i]

                scores = (
                    img_emb @ txt_emb.t()
                )  # shape = (num_images_per_sample x num_texts_per_sample)

                image_closest_text = scores.argmax(
                    dim=1
                )  # shape = (num_images_per_sample)
                text_closest_image = scores.argmax(
                    dim=0
                )  # shape = (num_texts_per_sample)
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
