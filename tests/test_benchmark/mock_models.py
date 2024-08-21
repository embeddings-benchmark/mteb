"""Mock models to be used for testing"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor

import mteb
from mteb.models.bge_models import BGEWrapper
from mteb.models.e5_models import E5Wrapper
from mteb.models.mxbai_models import MxbaiWrapper


class MockNumpyEncoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return np.random.rand(len(sentences), 10)


class MockTorchEncoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return torch.randn(len(sentences), 10)


class MockTorchbf16Encoder(mteb.Encoder):
    def __init__(self):
        pass

    def encode(self, sentences, prompt_name: str | None = None, **kwargs):
        return torch.randn(len(sentences), 10, dtype=torch.bfloat16)


class MockSentenceTransformer(SentenceTransformer):
    """A mock implementation of the SentenceTransformer intended to implement just the encode, method using the same arguments."""

    def __init__(self, *args, **kwargs):
        pass

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
        return torch.randn(len(sentences), 10)


class MockE5Wrapper(E5Wrapper):
    def __init__(self, **kwargs):
        self.mdl = MockSentenceTransformer()


class MockBGEWrapper(BGEWrapper):
    def __init__(self, **kwargs):
        self.mdl = MockSentenceTransformer()


class MockMxbaiWrapper(MxbaiWrapper):
    def __init__(self, **kwargs):
        self.mdl = MockSentenceTransformer()
