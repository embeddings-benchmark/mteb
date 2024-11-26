"""Mock models to be used for testing"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch import Tensor

import mteb
from mteb import SentenceTransformerWrapper
from mteb.encoder_interface import PromptType
from tests.test_benchmark.task_grid import MOCK_TASK_TEST_GRID


class MockNumpyEncoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return np.random.rand(len(sentences), 10)


class MockTorchEncoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return torch.randn(len(sentences), 10).numpy()


class MockTorchbf16Encoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return torch.randn(len(sentences), 10, dtype=torch.bfloat16)


class MockSentenceTransformer(SentenceTransformer):
    """A mock implementation of the SentenceTransformer intended to implement just the encode, method using the same arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"]
        | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = False,
    ) -> list[Tensor] | ndarray | Tensor:
        return torch.randn(len(sentences), 10).numpy()


class MockSentenceTransformerWrapper(SentenceTransformerWrapper):
    def __init__(
        self,
        model: str | SentenceTransformer | CrossEncoder,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for SentenceTransformer models.

        Args:
            model: The SentenceTransformer model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
            revision: The revision of the model to use.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        if isinstance(model, str):
            self.model = SentenceTransformer(
                model, revision=revision, trust_remote_code=True, **kwargs
            )
        else:
            self.model = model

        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            model_prompts = self.model.prompts
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            self.model.prompts = model_prompts
        self.model_prompts = model_prompts

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = get_mock_prompt_name(
                self.model_prompts, task_name, prompt_type
            )

        embeddings = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            **kwargs,  # sometimes in kwargs can be return_tensors=True
        )
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


def get_mock_prompt_name(
    task_to_prompt: dict[str, str], task_name: str, prompt_type: PromptType | None
) -> str | None:
    task = [
        mock_task
        for mock_task in MOCK_TASK_TEST_GRID
        if mock_task.metadata.name == task_name
    ][0]
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
    return None
