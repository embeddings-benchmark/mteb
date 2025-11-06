import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast, get_args, overload

from torch.utils.data import DataLoader

import mteb
from mteb.abstasks.task_metadata import TaskMetadata, TaskType
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

from .model_meta import ModelMeta, ScoringFunction

logger = logging.getLogger(__name__)


class AbsEncoder(ABC):
    """Base class to indicate that this is a wrapper for a model.

    Also contains some utility functions for wrappers for working with prompts and instructions.

    Attributes:
        model: The model to be wrapped.
        mteb_model_meta: Metadata about the model.
        model_prompts: A dictionary of prompts to be used for encoding sentences.
        instruction_template: A template for formatting instructions. Can be a string with '{instruction}' or
            a callable that takes the instruction and prompt type and returns a formatted instruction.
        prompts_dict: A dictionary of prompts to be used for encoding sentences, overrides model_prompts if provided.
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
            prompt_type: The prompt type (e.g. "query" | "document") to use for building the encoding prompt

        Returns:
            The name of the prompt to use, or None if no prompt is found.
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
        """Get the prompt to be used for encoding sentences.

        Args:
            task_metadata: The metadata of the task.
            prompt_type: The name type of prompt. (query or passage)
        """
        if not self.model_prompts:
            return None
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        return self.model_prompts.get(prompt_name)

    @staticmethod
    @overload
    def validate_task_to_prompt_name(
        task_to_prompt: dict[str, str] | None,
        raise_for_invalid_keys: Literal[True] = True,
    ) -> dict[str, str] | None: ...

    @staticmethod
    @overload
    def validate_task_to_prompt_name(
        task_to_prompt: dict[str, str] | None,
        raise_for_invalid_keys: Literal[False] = False,
    ) -> tuple[dict[str, str], Sequence[str]] | tuple[None, None]: ...

    @staticmethod
    def validate_task_to_prompt_name(
        task_to_prompt: dict[str, str] | None,
        raise_for_invalid_keys: bool = True,
    ) -> (
        dict[str, str] | tuple[dict[str, str], Sequence[str]] | tuple[None, None] | None
    ):
        """Validates that the keys in task_to_prompt_name map to a known task or prompt type.

        A key is valid if:

        1.  It is a valid task name; or
        2.  It is a valid task type; or
        3.  It is a valid prompt type; or
        4.  It is a compound key of the form "{task_name}-{prompt_type}" where task_name is a valid task type or task
            name and prompt_type is a valid prompt type.

        See the
        [MTEB docs](https://github.com/embeddings-benchmark/mteb/blob/main/docs/usage/usage.md#running-sentencetransformer-model-with-prompts)
        for a complete description of the order or precedence for these keys when running an evaluation.

        Arguments:
            task_to_prompt: The dictionary of prompts.
            raise_for_invalid_keys: If True, raise an error when an invalid key is encountered, otherwise return the
                list of error messages along with a filtered dictionary of prompts with valid keys. Defaults to True
                for backward compatibility.

        Returns:
            - None if `task_to_prompt` is None or empty;
            - Only a dictionary of validated prompts if `raise_for_invalid_keys` is `True`; or
            - A tuple containing the filtered dictionary of valid prompts and the set of error messages for the
                invalid prompts `raise_for_invalid` is `False`
        """
        if not task_to_prompt:
            return None if raise_for_invalid_keys else (None, None)

        task_types = get_args(TaskType)
        prompt_types = [e.value for e in PromptType]
        valid_keys_msg = f"Valid keys are task types [{task_types}], prompt types [{prompt_types}], and task names"
        valid_prompt_type_endings = tuple(
            f"-{prompt_type}" for prompt_type in prompt_types
        )

        invalid_keys: set[str] = set()
        invalid_task_messages: set[str] = set()

        for task_key in task_to_prompt:
            # task_key may be a compound key of the form "{task_name}-{prompt_type}". A task_name may contain a "-"
            # character (this occurs in ~12% of task names), so rsplit is used to separate a valid prompt_type postfix
            # from the unvalidated task_name.
            if task_key.endswith(valid_prompt_type_endings):
                task_name = task_key.rsplit("-", 1)[0]
            else:
                task_name = task_key

            if task_name not in task_types and task_name not in prompt_types:
                try:
                    mteb.get_task(task_name=task_name)
                except KeyError:
                    msg = f"Task name {task_name} is not valid. {valid_keys_msg}"
                    logger.warning(msg)
                    invalid_task_messages.add(msg)
                    invalid_keys.add(task_key)

        if raise_for_invalid_keys and invalid_task_messages:
            raise KeyError(invalid_task_messages)
        elif raise_for_invalid_keys:
            return task_to_prompt
        else:
            return {
                k: v for k, v in task_to_prompt.items() if k not in invalid_keys
            }, tuple(invalid_task_messages)

    def get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Get the instruction/prompt to be used for encoding sentences.

        Args:
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
                    6. Default prompt from the task definition
            prompt_type: The name type of prompt. (query or passage)

        Returns:
            The instruction/prompt to be used for encoding sentences.
        """
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
        """Format the instruction using the instruction template.

        Args:
            instruction: The instruction to be formatted.
            prompt_type: The name type of prompt. (query or passage)
        """
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
        """Create the instruction to be used for encoding sentences.

        Args:
            task_metadata: The metadata of the task
            prompt_type: The name type of prompt. (query or passage)

        Returns:
            The instruction to be used for encoding sentences.
        """
        instruction = self.get_instruction(task_metadata, prompt_type)
        if self.instruction_template and len(instruction) > 0:
            return self.format_instruction(instruction, prompt_type)
        return instruction

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the similarity between two collections of embeddings.

        The output will be a matrix with the similarity scores between all embeddings from the first parameter and all
        embeddings from the second parameter. This differs from similarity_pairwise which computes the similarity
        between corresponding pairs of embeddings.

        Read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity

        Args:
            embeddings1: [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2: [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.
        """
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
        """Compute the similarity between two collections of embeddings. The output will be a vector with the similarity scores between each pair of embeddings.

        Read more at: https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.similarity_pairwise

        Args:
            embeddings1: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2: [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            A [num_embeddings]-shaped torch tensor with pairwise similarity scores.
        """
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
