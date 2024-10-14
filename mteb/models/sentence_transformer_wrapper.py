from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, get_args

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb
from mteb.abstasks.TaskMetadata import TASK_TYPE
from mteb.encoder_interface import PromptType

logger = logging.getLogger(__name__)


class SentenceTransformerWrapper:
    def __init__(
        self,
        model: str | SentenceTransformer | CrossEncoder,
        revision: str | None = None,
        task_to_prompt_name: dict[str, str] | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(model, str):
            self.model = SentenceTransformer(
                model, revision=revision, trust_remote_code=True, **kwargs
            )
        else:
            self.model = model
        self.task_to_prompt_name = validate_task_to_prompt_name(task_to_prompt_name)
        if (
            hasattr(self.model, "prompts")
            and self.model.prompts is not None
            and task_to_prompt_name is not None
        ):
            if model_prompts is None:
                model_prompts = task_to_prompt_name
            logger.info(f"Model prompts will be overrided with {model_prompts}")
            self.model.prompts = model_prompts

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


        Returns:
            The encoded sentences.
        """
        prompt = None
        prompt_name = None
        if self.task_to_prompt_name is not None:
            prompt_name = get_prompt_name(
                self.task_to_prompt_name, task_name, prompt_type
            )
            logger.info(
                f"Using {prompt_name} prompt name for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")

        embeddings = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            **kwargs,  # sometimes in kwargs can be return_tensors=True
        )
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    def predict(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.predict(
            sentences,
            convert_to_numpy=True,
            **kwargs,
        )


def get_prompt_name(
    task_to_prompt: dict[str, str], task_name: str, prompt_type: PromptType | None
) -> str | None:
    """A wrapper function around the model.encode method that handles the prompt_name argument and standardizes the output to a numpy array.
    The order of priorities for prompt selection are:
        1. Composed prompt of task name + prompt type (query or passage)
        2. Specific task prompt
        3. Composed prompt of task type + prompt type (query or passage)
        4. Specific task type prompt
        5. Specific prompt type (query or passage)


    Args:
        task_to_prompt: The tasks names and their corresponding prompt_names
        task_name: The task name to use for building the encoding prompt
        prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt
    """
    import mteb

    task = mteb.get_task(task_name=task_name)
    task_type = task.metadata.type
    prompt_type_value = prompt_type.value if prompt_type else None

    if (
        task_name
        and prompt_type
        and f"{task_name}-{prompt_type_value}" in task_to_prompt
    ):
        return f"{task_name}-{prompt_type_value}"
    if task_name and task_name in task_to_prompt:
        return task_name
    if (
        task_type
        and prompt_type
        and f"{task_type}-{prompt_type_value}" in task_to_prompt
    ):
        return f"{task_type}-{prompt_type_value}"
    if task_type and task_type in task_to_prompt:
        return task_type
    if prompt_type and prompt_type in task_to_prompt:
        return prompt_type_value
    logger.info(
        "No combination of task name and prompt type was found in model prompts."
    )
    return None


def validate_task_to_prompt_name(task_to_prompt_name: dict[str, str]) -> dict[str, str]:
    task_types = get_args(TASK_TYPE)
    prompt_types = [e.value for e in PromptType]
    for task_name in task_to_prompt_name:
        if "-" in task_name:
            task_name, prompt_type = task_name.split("-")
            if prompt_type not in prompt_types:
                raise ValueError(
                    f"Prompt type {prompt_type} is not valid. Valid prompt types are {prompt_types}"
                )
        if task_name not in task_types and task_name not in prompt_types:
            task = mteb.get_task(task_name=task_name)
            if not task:
                raise ValueError(
                    f"Task name {task_name} is not valid. Valid task names are task types [{task_types}], prompt types [{prompt_types}] and task names"
                )
    return task_to_prompt_name
