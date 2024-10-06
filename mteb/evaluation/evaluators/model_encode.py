from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

import mteb
from mteb.encoder_interface import Encoder, PromptType

logger = logging.getLogger(__name__)


def model_encode(
    sentences: Sequence[str],
    *,
    model: Encoder,
    task_name: str | None,
    prompt_type: PromptType | None = None,
    **kwargs,
) -> np.ndarray:
    """A wrapper function around the model.encode method that handles the prompt_name argument and standardizes the output to a numpy array.
    The order of priorities for prompt selection are:
        1. Composed prompt of task name + prompt type (query or passage)
        2. Specific task prompt
        3. Composed prompt of task type + prompt type (query or passage)
        4. Specific task type prompt
        5. Specific prompt type (query or passage)


    Args:
        sentences: The sentences to encode
        model: The model to use for encoding
        task_name: The task name to use for building the encoding prompt
        prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt
        **kwargs: Additional arguments to pass to the model.encode method
    """
    if hasattr(model, "prompts"):
        task = mteb.get_task(task_name=task_name)
        task_type = task.metadata.type
        prompt_type_value = prompt_type.value if prompt_type else None

        # check if prompts is an empty dict
        if not model.prompts:  # type: ignore
            logger.info(
                "Model does not support prompts. Removing prompt_name argument."
            )
            kwargs.pop("prompt_name", None)

        if (
            task_name
            and prompt_type
            and f"{task_name}-{prompt_type_value}" in model.prompts
        ):
            kwargs["prompt_name"] = f"{task_name}-{prompt_type_value}"
        elif task_name and task_name in model.prompts:
            kwargs["prompt_name"] = task_name
        elif (
            task_type
            and prompt_type
            and f"{task_type}-{prompt_type_value}" in model.prompts
        ):
            kwargs["prompt_name"] = f"{task_type}-{prompt_type_value}"
        elif task_type and task_type in model.prompts:
            kwargs["prompt_name"] = task_type
        elif prompt_type and prompt_type in model.prompts:
            kwargs["prompt_name"] = prompt_type_value

        else:  # type: ignore
            logger.info(
                "No combination of task name and prompt type was found in model prompts. Removing prompt_name argument."
            )
            kwargs.pop("prompt_name", None)
    else:
        kwargs["prompt_name"] = task_name

    logger.info(
        f"Using {kwargs.get('prompt_name', None)} prompt name for "
        f"task={task_name} prompt_type={prompt_type}"
    )
    logger.info(f"Encoding {len(sentences)} sentences.")

    embeddings = model.encode(sentences, **kwargs)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().float()

    return np.asarray(embeddings)
