from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from mteb.encoder_interface import PromptType

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class Model2VecWrapper(Wrapper):
    def __init__(
        self,
        model: str,
        **kwargs,
    ) -> None:
        """Wrapper for Model2Vec models.

        Args:
            model: The Model2Vec model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
        """
        self.model = model
        self.static_model = StaticModel.from_pretrained(self.model)

    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        return self.static_model.encode(sentences)
