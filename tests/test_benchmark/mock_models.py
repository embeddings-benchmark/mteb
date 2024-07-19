"""Mock models to be used for testing"""

from __future__ import annotations

import numpy as np
import torch

import mteb


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
