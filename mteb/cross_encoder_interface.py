from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import torch

Corpus = Union[list[dict[str, str]], dict[str, list[str]]]


@runtime_checkable
class CrossEncoder(Protocol):
    """The interface for a cross-encoder in MTEB.

    In general the interface is kept aligned with sentence-transformers interface. In cases where exceptions occurs these are handled within MTEB.
    """

    def __init__(self, device: str | None = None) -> None:
        """The initialization function for the cross-encoder. Used when calling it from the mteb run CLI.

        Args:
            device: The device to use for prediction. Can be ignored if the encoder is not using a device (e.g. for API)
        """

    def predict(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
        *,
        task_name: str | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Predicts scores for query-passage pairs. Note that unlike the encoder, the prompt is specified in the model as it is typically more complex and model-specific.

        Args:
            queries: The queries to score.
            passages: The passages to score.
            instruction: Optional instruction text to combine with the query.
            **kwargs: Additional arguments to pass to the cross-encoder.

        Returns:
            The predicted scores for each query-passage pair.
        """
        ...
