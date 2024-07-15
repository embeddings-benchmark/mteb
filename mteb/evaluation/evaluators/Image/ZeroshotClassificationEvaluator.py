from __future__ import annotations

import logging
from typing import Any

from PIL import Image
from sklearn import metrics

from mteb.encoder_interface import Encoder

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ZeroshotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        images: list[Image.Image],
        labels: list[int],
        candidate_labels: list[str],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.images = images
        self.labels = labels
        self.candidate_labels = candidate_labels
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        text_embeddings = model.get_text_embeddings(
            self.candidate_labels, batch_size=encode_kwargs["batch_size"]
        )
        image_embeddings = model.get_image_embeddings(
            self.images, batch_size=encode_kwargs["batch_size"]
        )
        probs = model.calculate_probs(text_embeddings, image_embeddings)
        predictions = probs.argmax(dim=1)

        logger.info("Evaluating...")
        accuracy = metrics.accuracy_score(self.labels, predictions)

        return {"accuracy": accuracy}
