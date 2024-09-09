from __future__ import annotations

import itertools
import logging
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


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
        images: list[list[Image.Image]],
        texts: list[list[str]],
        task_name: str | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            images = images[:limit]
            texts = texts[:limit]
        self.images = images
        self.texts = texts
        self.task_name = task_name

        assert len(self.images) == len(self.texts)

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 64

        num_samples = len(self.images)
        num_images_per_sample = len(self.images[0])
        num_texts_per_sample = len(self.texts[0])

        images = list(itertools.chain.from_iterable(self.images))
        texts = list(itertools.chain.from_iterable(self.texts))

        image_embeddings = F.normalize(
            model.get_image_embeddings(images, batch_size=encode_kwargs["batch_size"]),
            dim=-1,
        ).view(num_samples, num_images_per_sample, -1)
        text_embeddings = F.normalize(
            model.get_text_embeddings(texts, batch_size=encode_kwargs["batch_size"]),
            dim=-1,
        ).view(num_samples, num_texts_per_sample, -1)
        img_ground_truths = torch.arange(num_images_per_sample)
        caption_ground_truths = torch.arange(num_texts_per_sample)

        image_score = []
        text_score = []
        score = []

        for i in range(num_samples):
            images_emb = image_embeddings[i]
            texts_emb = text_embeddings[i]
            scores = (
                images_emb @ texts_emb.t()
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
