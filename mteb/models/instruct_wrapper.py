from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch

from mteb.encoder_interface import PromptType

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


def instruct_wrapper(
    model_name_or_path: str,
    mode: str,
    instruction_template: str | Callable[[str], str] | None = None,
    **kwargs,
):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            f"Please install `pip install gritlm` to use {model_name_or_path}."
        )

    class InstructWrapper(GritLM, Wrapper):
        def __init__(
            self,
            model_name_or_path: str,
            mode: str,
            instruction_template: str | Callable[[str], str] | None = None,
            **kwargs,
        ):
            if (
                isinstance(instruction_template, str)
                and "{instruction}" not in instruction_template
            ):
                raise ValueError(
                    "Instruction template must contain the string '{instruction}'."
                )
            if instruction_template is None:
                logger.warning(
                    "No instruction template provided. Instructions will be used as-is."
                )

            if "gte-Qwen" in model_name_or_path:
                logger.warning(
                    "Instructions are used in both query and docs, which may cause performance discrepancies from the original implementation."
                )

            self.instruction_template = instruction_template
            super().__init__(model_name_or_path=model_name_or_path, mode=mode, **kwargs)

        def encode(
            self,
            sentences: Sequence[str],
            *args,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            instruction = self.get_instruction(task_name, prompt_type)

            if self.instruction_template:
                instruction = self.format_instruction(instruction, prompt_type)

            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")
            embeddings = super().encode(
                sentences, instruction=instruction, *args, **kwargs
            )
            if isinstance(embeddings, torch.Tensor):
                # sometimes in kwargs can be return_tensors=True
                embeddings = embeddings.cpu().detach().float().numpy()
            return embeddings

    return InstructWrapper(model_name_or_path, mode, instruction_template, **kwargs)
