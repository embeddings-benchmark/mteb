from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Literal, get_args, overload

import mteb
from mteb.abstasks.TaskMetadata import TASK_TYPE
from mteb.encoder_interface import PromptType

logger = logging.getLogger(__name__)


class Wrapper:
    """Base class to indicate that this is a wrapper for a model.
    Also contains some utility functions for wrappers for working with prompts and instructions.
    """

    instruction_template: str | Callable[[str, str], str] | None = None

    @staticmethod
    def get_prompt_name(
        task_to_prompt: dict[str, str] | None,
        task_name: str,
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
            task_to_prompt: The tasks names and their corresponding prompt_names
            task_name: The task name to use for building the encoding prompt
            prompt_type: The prompt type (e.g. "query" | "document") to use for building the encoding prompt
        """
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
        if prompt_type and prompt_type_value in task_to_prompt:
            return prompt_type_value
        logger.info(
            "No combination of task name and prompt type was found in model prompts."
        )
        return None

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
            *   None if `task_to_prompt` is None or empty;
            *   Only a dictionary of validated prompts if `raise_for_invalid_keys` is `True`; or
            *   A tuple continaing the filtered dictionary of valid prompts and the set of error messages for the
                invalid prompts `raise_for_invalid` is `False`

        Raises:
            KeyError: If any invlaid keys are encountered and `raise_for_invalid_keys` is `True`, this function will
            raise a single `KeyError` contianing the
        """
        if not task_to_prompt:
            return None if raise_for_invalid_keys else (None, None)

        task_types = get_args(TASK_TYPE)
        prompt_types = [e.value for e in PromptType]
        valid_keys_msg = f"Valid keys are task types [{task_types}], prompt types [{prompt_types}], and task names"
        valid_prompt_type_endings = tuple(
            [f"-{prompt_type}" for prompt_type in prompt_types]
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

    @staticmethod
    def get_instruction(
        task_name: str,
        prompt_type: PromptType | None,
        prompts_dict: dict[str, str] | None = None,
    ) -> str:
        """Get the instruction/prompt to be used for encoding sentences."""
        task = mteb.get_task(task_name=task_name)
        task_metadata = task.metadata
        prompt = task_metadata.prompt
        if prompts_dict and task_name in prompts_dict:
            prompt = prompts_dict[task_name]

        if isinstance(prompt, dict) and prompt_type:
            if prompt.get(prompt_type.value):
                return prompt[prompt_type.value]
            logger.warning(
                f"Prompt type '{prompt_type}' not found in task metadata for task '{task_name}'."
            )
            return ""

        if prompt:
            return prompt
        return task.abstask_prompt

    def format_instruction(
        self, instruction: str, prompt_type: PromptType | None = None
    ) -> str:
        if isinstance(self.instruction_template, str):
            if "{instruction}" not in self.instruction_template:
                raise ValueError(
                    "Instruction template must contain the string '{instruction}'."
                )
            return self.instruction_template.format(instruction=instruction)
        return self.instruction_template(instruction, prompt_type)

    def get_task_instruction(
        self,
        task_name: str,
        prompt_type: PromptType | None,
        prompts_dict: dict[str, str] | None = None,
    ) -> str:
        instruction = self.get_instruction(task_name, prompt_type, prompts_dict)
        if self.instruction_template:
            return self.format_instruction(instruction)
        return instruction
