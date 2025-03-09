from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.create_dataloaders import (
    convert_images_to_rgb,
    create_image_dataloader,
)
from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


def make_custom_collate_fn(
    images_column_names: str | list[str],
    texts_column_names: str | list[str],
    encode_image: bool = False,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Factory function to create a collate_fn that mimics the behavior of
    ImageTextDataset.__getitem__. For each sample in the batch, it extracts the images
    and texts according to the provided column names and applies an optional transform
    to the images.

    Args:
        images_column_names: A column name (str) or list of column names for images.
        texts_column_names: A column name (str) or list of column names for texts.
        encode_image: If encode images or not

    Returns:
        A collate_fn that can be passed to a DataLoader.
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_texts = []
        collated_images = []
        for data in batch:
            # Extract images from the sample.
            if isinstance(images_column_names, str):
                images = data[images_column_names]
            else:
                images = [data[col] for col in images_column_names]

            # Extract texts from the sample.
            if isinstance(texts_column_names, str):
                texts = data[texts_column_names]
            else:
                texts = [data[col] for col in texts_column_names]

            collated_images.extend(images)
            collated_texts.extend(texts)
        if encode_image:
            return {"image": collated_images}
        return {
            "text": collated_texts,
        }

    return collate_fn


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
        self.dataset = dataset
        self.images_column_names = images_column_names
        self.texts_column_names = texts_column_names
        self.task_name = task_name

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

        if isinstance(self.images_column_names, list):
            for img_column in tqdm(
                self.images_column_names, desc="Transforming images to RGB"
            ):
                self.dataset = self.dataset.map(
                    convert_images_to_rgb,
                    fn_kwargs={"image_col_name": img_column},
                    desc="Transforming images to RGB",
                    num_proc=4,
                )
        else:
            self.dataset = self.dataset.map(
                convert_images_to_rgb,
                fn_kwargs={"image_col_name": self.images_column_names},
                desc="Transforming images to RGB",
            )

        img_ground_truths = torch.arange(num_images_per_sample)
        caption_ground_truths = torch.arange(num_texts_per_sample)

        text_embeddings = model.encode(
            DataLoader(
                self.dataset,
                batch_size=encode_kwargs["batch_size"],
                collate_fn=make_custom_collate_fn(
                    self.images_column_names,
                    self.texts_column_names,
                    encode_image=False,
                ),
            ),
            task_name=self.task_name,
            is_image_encode=False,
            **encode_kwargs,
        )

        norm_text_embeddings = F.normalize(
            text_embeddings,
            dim=-1,
        ).view(len(self.dataset), num_texts_per_sample, -1)

        image_embeddings = model.encode(
            create_image_dataloader(
                self.dataset,
                batch_size=encode_kwargs["batch_size"],
                collate_fn=make_custom_collate_fn(
                    self.images_column_names,
                    self.texts_column_names,
                    encode_image=True,
                ),
            ),
            task_name=self.task_name,
            is_image_encode=True,
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
