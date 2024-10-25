from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch

from mteb.encoder_interface import PromptType

from .wrapper import Wrapper


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
            "Please install `pip install gritlm` to use gte-Qwen2-7B-instruct."
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
                instruction = self.format_instruction(instruction)
            embeddings = super().encode(
                sentences, instruction=instruction, *args, **kwargs
            )
            if isinstance(embeddings, torch.Tensor):
                # sometimes in kwargs can be return_tensors=True
                embeddings = embeddings.cpu().detach().float().numpy()
            return embeddings

        def format_instruction(self, instruction: str) -> str:
            if isinstance(self.instruction_template, str):
                return self.instruction_template.format(instruction=instruction)
            return self.instruction_template(instruction)

    return InstructWrapper(model_name_or_path, mode, instruction_template, **kwargs)
