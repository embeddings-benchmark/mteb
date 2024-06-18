from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mteb.encoder_interface import Encoder

logger = logging.getLogger(__name__)


def model_encode(
    sentences: Sequence[str], *, model: Encoder, prompt_name: str | None, **kwargs
) -> np.ndarray:
    kwargs["prompt_name"] = prompt_name
    if hasattr(model, "prompts") and prompt_name not in model.prompts:  # type: ignore
        logger.info(
            f"Prompt {prompt_name} not found in model prompts. Removing prompt_name argument."
        )
        kwargs.pop("prompt_name")

    logger.info(f"Encoding {len(sentences)} sentences.")

    embeddings = model.encode(sentences, **kwargs)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach()

    return np.asarray(embeddings)
