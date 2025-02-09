from __future__ import annotations

import logging
from typing import Callable, get_args

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
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt
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
    def validate_task_to_prompt_name(
        task_to_prompt_name: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if task_to_prompt_name is None:
            return task_to_prompt_name
        task_types = get_args(TASK_TYPE)
        prompt_types = [e.value for e in PromptType]
        for task_name in task_to_prompt_name:
            if "-" in task_name:
                task_name, prompt_type = task_name.split("-")
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
        return task_to_prompt_name

    @staticmethod
    def get_instruction(task_name: str, prompt_type: PromptType | None) -> str:
        """Get the instruction/prompt to be used for encoding sentences."""
        task = mteb.get_task(task_name=task_name)
        task_metadata = task.metadata
        if isinstance(task_metadata.prompt, dict) and prompt_type:
            if task_metadata.prompt.get(prompt_type.value):
                return task_metadata.prompt[prompt_type.value]
            logger.warning(
                f"Prompt type '{prompt_type}' not found in task metadata for task '{task_name}'."
            )
            return ""
        if task_metadata.prompt:
            return task_metadata.prompt
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
        self, task_name: str, prompt_type: PromptType | None
    ) -> str:
        instruction = self.get_instruction(task_name, prompt_type)
        if self.instruction_template:
            return self.format_instruction(instruction)
        return instruction
