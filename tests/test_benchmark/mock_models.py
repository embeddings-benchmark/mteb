"""Mock models to be used for testing"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch import Tensor
from torch.utils.data import DataLoader

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.load_results.task_results import Namespace
from mteb.model_meta import ModelMeta
from mteb.models import BaseEncoder, SentenceTransformerWrapper
from mteb.types import Array, BatchedInput, PromptType


class BaseMockEncoder(BaseEncoder):
    def __init__(self):
        pass


class MockNumpyEncoder(BaseMockEncoder):
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
        return np.random.rand(len(inputs.dataset), 10)  # type: ignore


class MockTorchEncoder(BaseMockEncoder):
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
        return torch.randn(len(inputs.dataset), 10)


class MockTorchfp16Encoder(BaseMockEncoder):
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
        return torch.randn(len(inputs.dataset), 10, dtype=torch.float16)  # type: ignore


class MockSentenceTransformersbf16Encoder(SentenceTransformer):
    """Ensure that data types not supported by the encoder are converted to the supported data type."""

    model_card_data = Namespace(
        model_name="MockSentenceTransformersbf16Encoder", base_model_revision="1.0.0"
    )

    def __init__(self):
        pass

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
        return torch.randn(len(inputs), 10, dtype=torch.bfloat16)  # type: ignore

    @staticmethod
    def get_sentence_embedding_dimension() -> int:
        return 10


class MockCLIPEncoder(BaseMockEncoder):
    mteb_model_meta = ModelMeta(
        loader=None,
        name="mock/MockCLIPModel",
        languages=["eng_Latn"],
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        release_date="2021-02-06",
        modalities=["image", "text"],
        n_parameters=86_600_000,
        memory_usage_mb=330,
        max_tokens=None,
        embed_dim=768,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        reference="https://huggingface.co/openai/clip-vit-base-patch32",
        similarity_fn_name=None,
        use_instructions=False,
        training_datasets=None,
    )
    model_card_data = mteb_model_meta

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
        return torch.randn(len(inputs.dataset), 10)


class MockMocoEncoder(BaseMockEncoder):
    mteb_model_meta = ModelMeta(
        loader=None,
        name="mock/MockMocoModel",
        languages=["eng_Latn"],
        revision="7d091cd70772c5c0ecf7f00b5f12ca609a99d69d",
        release_date="2024-01-01",
        modalities=["image"],
        n_parameters=86_600_000,
        memory_usage_mb=330,
        max_tokens=None,
        embed_dim=768,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        reference="https://github.com/facebookresearch/moco-v3",
        similarity_fn_name=None,
        use_instructions=False,
        training_datasets=None,
    )


class MockSentenceTransformer(SentenceTransformer):
    """A mock implementation of the SentenceTransformer intended to implement just the encode, method using the same arguments."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(
        self,
        sentences: list[str],
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
        **kwargs: Any,
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
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)

        embeddings = self.model.encode(
            inputs,
            prompt_name=prompt_name,
            **kwargs,  # sometimes in kwargs can be return_tensors=True
        )
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings
