from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Encoder(Protocol):
    """The interface for an encoder in MTEB."""

    def encode(
        self, sentences: list[str], prompt: str, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            prompt: The prompt to use. Useful for prompt-based models.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        ...


class EncoderWithQueryCorpusEncode(Encoder, Protocol):
    """The interface for an encoder that supports encoding a queries and a corpus."""

    def encode_queries(
        self, queries: list[str], prompt: str, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given queries using the encoder.

        Args:
            queries: The queries to encode.
            prompt: The prompt to use. Useful for prompt-based models.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded queries.
        """
        ...

    def encode_corpus(
        self, corpus: list[str], prompt: str, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given corpus using the encoder.

        Args:
            corpus: The corpus to encode.
            prompt: The prompt to use. Useful for prompt-based models.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded corpus.
        """
        ...
