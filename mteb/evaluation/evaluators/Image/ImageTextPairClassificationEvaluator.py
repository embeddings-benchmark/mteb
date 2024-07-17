from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

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
        sentences1: The first column of sentences
        sentences2: The second column of sentences
        labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
        name: Name for the output
        batch_size: Batch size used to compute embeddings
        write_csv: Write results to a CSV file
    """

    def __init__(
        self,
        image0,
        image1,
        text0,
        text1,
        task_name: str | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            image0 = image0[:limit]
            image1 = image1[:limit]
            text0 = text0[:limit]
            text1 = text1[:limit]
        self.image0 = image0
        self.image1 = image1
        self.text0 = text0
        self.text1 = text1
        self.task_name = task_name

        assert len(self.image0) == len(self.image1)
        assert len(self.text0) == len(self.text1)
        assert len(self.image0) == len(self.text0)

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        images = self.image0 + self.image1
        texts = self.text0 + self.text1
        num_images = len(self.image0)

        image_embeddings = F.normalize(
            model.get_image_embeddings(images, batch_size=encode_kwargs["batch_size"]),
            dim=-1,
        ).view(num_images, 2, -1)
        text_embeddings = F.normalize(
            model.get_text_embeddings(texts, batch_size=encode_kwargs["batch_size"]),
            dim=-1,
        ).view(num_images, 2, -1)
        ground_truths = torch.tensor([0, 1])

        image_score = []
        text_score = []
        score = []

        for i in range(num_images):
            images_emb = image_embeddings[i]
            texts_emb = text_embeddings[i]
            scores = images_emb @ texts_emb.t()

            image_closest_text = scores.argmax(dim=1)
            text_closest_image = scores.argmax(dim=0)
            pred_text_is_correct = (image_closest_text == ground_truths).all().item()
            pred_image_is_correct = (text_closest_image == ground_truths).all().item()
            all_correct = pred_text_is_correct and pred_image_is_correct
            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            score.append(all_correct)

        metrics = {}
        # metrics["image_acc"] = torch.Tensor(image_score).float().mean().item()
        # metrics["text_acc"] = torch.Tensor(text_score).float().mean().item()
        metrics["accuracy"] = torch.Tensor(score).float().mean().item()
        return metrics
