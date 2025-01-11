from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import mteb
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
            f"Please install `pip install mteb[gritlm]` to use {model_name_or_path}."
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


class InstructSentenceTransformerWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str | Callable[[str], str] | None = None,
        max_seq_length: int | None = None,
        apply_instruction_to_passages: bool = True,
        **kwargs: Any,
    ):
        """Args:
        model_name: model name
        revision: revision
        instruction_template: Model template. Should contain the string '{instruction}'.
        max_seq_length: maximum sequence length. If None, the maximum sequence length won't be changed.
        apply_instruction_to_passages: Whether to apply the instruction template to the passages.
        **kwargs:
        """
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

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_passages
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        task = mteb.get_task(task_name=task_name)
        instruction = self.get_instruction(task_name, prompt_type)
        if self.instruction_template:
            instruction = self.format_instruction(instruction, prompt_type)

        # to passage prompts won't be applied to passages
        if not self.apply_instruction_to_passages:
            if task.metadata.type == "p2p":
                instruction = None
            elif prompt_type == PromptType.passage and task.metadata.type == "s2p":
                instruction = None

        logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")
        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings
