from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mteb.encoder_interface import Encoder

logger = logging.getLogger(__name__)


def model_encode(
    sentences: Sequence[str], *, model: Encoder, task_name: str | None, **kwargs
) -> np.ndarray:
    if hasattr(model, "prompts") and task_name in model.prompts:  # type: ignore
        logger.info(
            f"Prompt {task_name} found in model prompts. Adding prompt_name argument."
        )
        kwargs["prompt_name"] = task_name

    logger.info(f"Encoding {len(sentences)} sentences.")

    embeddings = model.encode(sentences, **kwargs)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach()

    return np.asarray(embeddings)
