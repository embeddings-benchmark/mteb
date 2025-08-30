from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, get_args

from torch.utils.data import DataLoader

import mteb
from mteb.abstasks.task_metadata import TaskMetadata, TaskType
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import (
    cos_sim,
    dot_score,
    max_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_max_sim,
)
from mteb.types import (
    Array,
    BatchedInput,
    PromptType,
)

logger = logging.getLogger(__name__)


class AbsEncoder(ABC):
    """Base class to indicate that this is a wrapper for a model.
    Also contains some utility functions for wrappers for working with prompts and instructions.
    """

    model: Any
    mteb_model_meta: ModelMeta | None = None
    model_prompts: dict[str, str] | None = None
    instruction_template: str | Callable[[str, PromptType], str] | None = None
    prompts_dict: dict[str, str] | None = None

    def get_prompt_name(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        """A wrapper function around the model.encode method that handles the prompt_name argument and standardizes the output to a numpy array.
        The order of priorities for prompt selection are:
            1. Composed prompt of task name + prompt type (query or passage)
            2. Specific task prompt
            3. Composed prompt of task type + prompt type (query or passage)
            4. Specific task type prompt
            5. Specific prompt type (query or passage)


        Args:
            task_metadata: The task name to use for building the encoding prompt
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt
        """
        if self.model_prompts is None:
            return None
        task_type = task_metadata.type
        prompt_type_value = prompt_type.value if prompt_type else None
        task_name = task_metadata.name

        if (
            task_name
            and prompt_type
            and f"{task_name}-{prompt_type_value}" in self.model_prompts
        ):
            return f"{task_name}-{prompt_type_value}"
        if task_name and task_name in self.model_prompts:
            return task_name
        if (
            task_type
            and prompt_type
            and f"{task_type}-{prompt_type_value}" in self.model_prompts
        ):
            return f"{task_type}-{prompt_type_value}"
        if task_type and task_type in self.model_prompts:
            return task_type
        if prompt_type and prompt_type_value in self.model_prompts:
            return prompt_type_value
        logger.info(
            "No combination of task name and prompt type was found in model prompts."
        )
        return None

    def get_prompt(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        if not self.model_prompts:
            return None
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        return self.model_prompts.get(prompt_name)  # type: ignore

    def validate_task_to_prompt_name(self) -> None:
        """Validate the task name and prompt type against the model prompts.

        All keys in model_prompts should be valid task names, prompt types or the combination of both.
        """
        if self.model_prompts is None:
            return
        task_types = get_args(TaskType)
        prompt_types = [e.value for e in PromptType]
        for task_name in self.model_prompts:
            if "-" in task_name and task_name.endswith(
                (f"-{PromptType.query.value}", f"-{PromptType.document.value}")
            ):
                task_name, prompt_type = task_name.rsplit("-", 1)
                if prompt_type not in prompt_types:
                    msg = f"Prompt type {prompt_type} is not valid. Valid prompt types are {prompt_types}"
                    logger.warning(msg)
                    raise KeyError(msg)
            if task_name not in task_types and task_name not in prompt_types:
                task = mteb.get_task(task_name=task_name)
                if not task:
                    msg = f"Task name {task_name} is not valid. Valid task names are task types [{task_types}], prompt types [{prompt_types}] and task names"
                    logger.warning(msg)
                    raise KeyError(msg)

    def get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Get the instruction/prompt to be used for encoding sentences."""
        prompt = task_metadata.prompt
        if self.prompts_dict and task_metadata.name in self.prompts_dict:
            prompt = self.prompts_dict[task_metadata.name]

        if isinstance(prompt, dict) and prompt_type:
            if prompt.get(prompt_type.value):
                return prompt[prompt_type.value]
            logger.warning(
                f"Prompt type '{prompt_type}' not found in task metadata for task '{task_metadata.name}'."
            )
            return ""

        if prompt:
            return prompt

        abstask = mteb.get_task(task_name=task_metadata.name)
        return abstask.abstask_prompt

    def format_instruction(
        self, instruction: str, prompt_type: PromptType | None = None
    ) -> str:
        if self.instruction_template is None:
            raise ValueError(
                "Attempting to format an instruction without an instruction template."
            )
        if isinstance(self.instruction_template, str):
            if "{instruction}" not in self.instruction_template:
                raise ValueError(
                    "Instruction template must contain the string '{instruction}'."
                )
            return self.instruction_template.format(instruction=instruction)
        return self.instruction_template(instruction, prompt_type)

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        instruction = self.get_instruction(task_metadata, prompt_type)
        if self.instruction_template and len(instruction) > 0:
            return self.format_instruction(instruction)
        return instruction

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        if self.mteb_model_meta is None or (
            self.mteb_model_meta is not None
            and self.mteb_model_meta.similarity_fn_name is None
        ):
            if (
                hasattr(self, "model")
                and hasattr(self.model, "similarity")
                and callable(self.model.similarity)
            ):
                arr = self.model.similarity(embeddings1, embeddings2)
                # We assume that the model returns an Array-like object:
                arr = cast(Array, arr)
                return arr
            return cos_sim(embeddings1, embeddings2)
        if self.mteb_model_meta.similarity_fn_name is ScoringFunction.COSINE:
            return cos_sim(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT:
            return dot_score(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.MAX_SIM:
            return max_sim(embeddings1, embeddings2)
        raise ValueError("Similarity function not specified.")

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        if self.mteb_model_meta is None or (
            self.mteb_model_meta is not None
            and self.mteb_model_meta.similarity_fn_name is None
        ):
            if (
                hasattr(self, "model")
                and hasattr(self.model, "similarity_pairwise")
                and callable(self.model.similarity_pairwise)
            ):
                arr = self.model.similarity_pairwise(embeddings1, embeddings2)
                # We assume that the model returns an Array-like object:
                arr = cast(Array, arr)
                return arr
            return pairwise_cos_sim(embeddings1, embeddings2)
        if self.mteb_model_meta.similarity_fn_name is ScoringFunction.COSINE:
            return pairwise_cos_sim(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT:
            return pairwise_dot_score(embeddings1, embeddings2)
        elif self.mteb_model_meta.similarity_fn_name is ScoringFunction.MAX_SIM:
            return pairwise_max_sim(embeddings1, embeddings2)
        raise ValueError("Similarity function not specified.")

    @abstractmethod
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encodes the given sentences using the encoder.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        raise NotImplementedError(
            "The encode method must be implemented in the subclass."
        )
