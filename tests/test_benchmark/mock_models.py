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


class MockCLIPEncoder:
    def __init__(self):
        pass

    def get_text_embeddings(self, texts, **kwargs):
        return torch.randn(len(texts), 10)

    def get_image_embeddings(self, images, **kwargs):
        return torch.randn(len(images), 10)

    def get_fused_embeddings(self, texts, images, **kwargs):
        return torch.randn(len(images), 10)

    def calculate_probs(self, text_embeddings, image_embeddings):
        return torch.randn(image_embeddings.shape[0], text_embeddings.shape[1])
