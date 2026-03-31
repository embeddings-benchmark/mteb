from __future__ import annotations

import sys
import types

import pytest
import torch
from PIL import Image

from mteb import get_model_meta
from mteb.models.model_implementations import colmodernvbert_models, colpali_models
from mteb.models.model_implementations.colmodernvbert_models import (
    BiModernVBertWrapper,
    ColModernVBertWrapper,
)
from mteb.models.model_meta import ScoringFunction


class _FakeDataset:
    def __init__(self, features: dict[str, object]):
        self.features = features


class _FakeLoader:
    def __init__(self, batches: list[dict[str, object]], features: dict[str, object]):
        self._batches = batches
        self.dataset = _FakeDataset(features)

    def __iter__(self):
        return iter(self._batches)


class _FakeModel:
    def __init__(self):
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        if "pixel_values" in inputs:
            batch_size = inputs["pixel_values"].shape[0]
            return torch.full((batch_size, 2, 3), 2.0)
        batch_size = inputs["input_ids"].shape[0]
        return torch.full((batch_size, 2, 3), 1.0)


class _FakeProcessor:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def process_texts(self, texts: list[str]):
        self.calls.append(("texts", list(texts)))
        return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

    def process_images(self, images: list[Image.Image]):
        self.calls.append(("images", len(images)))
        return {"pixel_values": torch.ones((len(images), 3, 2, 2))}

    def score(self, a, b, device=None):
        self.calls.append(("score", device))
        return torch.tensor([[0.5]])


class _FakeBiModel:
    def __init__(self):
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        batch_size = next(iter(inputs.values())).shape[0]
        if "pixel_values" in inputs:
            return torch.full((batch_size, 4), 2.0)
        return torch.full((batch_size, 4), 1.0)


class _TaskMetadataStub:
    name = "mock-task"
    type = "Retrieval"
    prompt = None


@pytest.fixture
def colmodernvbert_wrapper(monkeypatch):
    fake_engine = types.ModuleType("colpali_engine")
    fake_models = types.ModuleType("colpali_engine.models")
    fake_models.ColModernVBert = _FakeModel
    fake_models.ColModernVBertProcessor = _FakeProcessor
    fake_models.BiModernVBert = _FakeBiModel
    fake_models.BiModernVBertProcessor = _FakeProcessor
    fake_engine.models = fake_models

    monkeypatch.setitem(sys.modules, "colpali_engine", fake_engine)
    monkeypatch.setitem(sys.modules, "colpali_engine.models", fake_models)
    monkeypatch.setattr(colmodernvbert_models, "requires_package", lambda *a, **k: None)
    monkeypatch.setattr(
        colmodernvbert_models, "requires_image_dependencies", lambda *a, **k: None
    )
    monkeypatch.setattr(colpali_models, "requires_package", lambda *a, **k: None)
    monkeypatch.setattr(
        colpali_models, "requires_image_dependencies", lambda *a, **k: None
    )

    return ColModernVBertWrapper(device="cpu")


def test_colmodernvbert_text_and_image_encoding(colmodernvbert_wrapper):
    text_loader = _FakeLoader(
        batches=[{"text": ["first", "second"]}],
        features={"text": object()},
    )
    image_loader = _FakeLoader(
        batches=[{"image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]}],
        features={"image": object()},
    )
    mixed_loader = _FakeLoader(
        batches=[
            {
                "text": ["first", "second"],
                "image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
            }
        ],
        features={"text": object(), "image": object()},
    )

    text_embeddings = colmodernvbert_wrapper.get_text_embeddings(text_loader)
    image_embeddings = colmodernvbert_wrapper.get_image_embeddings(image_loader)
    fused_embeddings = colmodernvbert_wrapper.encode(
        mixed_loader,
        task_metadata=_TaskMetadataStub(),
        hf_split="test",
        hf_subset="default",
    )

    assert text_embeddings.shape == (2, 2, 3)
    assert image_embeddings.shape == (2, 2, 3)
    assert torch.equal(fused_embeddings, text_embeddings + image_embeddings)
    assert ("texts", ["first", "second"]) in colmodernvbert_wrapper.processor.calls
    assert ("images", 2) in colmodernvbert_wrapper.processor.calls


def test_colmodernvbert_mixed_length_mismatch_raises(colmodernvbert_wrapper):
    mixed_loader = _FakeLoader(
        batches=[
            {
                "text": ["first", "second"],
                "image": [Image.new("RGB", (2, 2))],
            }
        ],
        features={"text": object(), "image": object()},
    )

    with pytest.raises(ValueError, match="same length"):
        colmodernvbert_wrapper.encode(
            mixed_loader,
            task_metadata=_TaskMetadataStub(),
            hf_split="test",
            hf_subset="default",
        )


def test_colmodernvbert_similarity_delegates_to_processor(colmodernvbert_wrapper):
    scores = colmodernvbert_wrapper.similarity(
        torch.ones((1, 2, 3)), torch.ones((1, 2, 3))
    )

    assert torch.equal(scores, torch.tensor([[0.5]]))
    assert ("score", "cpu") in colmodernvbert_wrapper.processor.calls


def test_colmodernvbert_model_meta_is_registered():
    model_meta = get_model_meta("ModernVBERT/colmodernvbert")

    assert model_meta.loader is ColModernVBertWrapper
    assert model_meta.revision == "e1e601df2542530091ade8a7b43c0bee99b58432"


def test_bimodernvbert_text_and_image_encoding(monkeypatch):
    monkeypatch.setattr(
        colmodernvbert_models, "requires_package", lambda *a, **k: None
    )
    monkeypatch.setattr(
        colmodernvbert_models, "requires_image_dependencies", lambda *a, **k: None
    )
    bi_wrapper = BiModernVBertWrapper(device="cpu")

    text_loader = _FakeLoader(
        batches=[{"text": ["first", "second"]}],
        features={"text": object()},
    )
    image_loader = _FakeLoader(
        batches=[{"image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]}],
        features={"image": object()},
    )
    mixed_loader = _FakeLoader(
        batches=[
            {
                "text": ["first", "second"],
                "image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
            }
        ],
        features={"text": object(), "image": object()},
    )

    text_embeddings = bi_wrapper.get_text_embeddings(text_loader)
    image_embeddings = bi_wrapper.get_image_embeddings(image_loader)
    fused_embeddings = bi_wrapper.encode(
        mixed_loader,
        task_metadata=_TaskMetadataStub(),
        hf_split="test",
        hf_subset="default",
    )

    assert text_embeddings.shape == (2, 4)
    assert image_embeddings.shape == (2, 4)
    assert torch.equal(fused_embeddings, text_embeddings + image_embeddings)
    assert ("texts", ["first", "second"]) in bi_wrapper.processor.calls
    assert ("images", 2) in bi_wrapper.processor.calls


def test_bimodernvbert_model_meta_is_registered():
    model_meta = get_model_meta("ModernVBERT/bimodernvbert")

    assert model_meta.loader is BiModernVBertWrapper
    assert model_meta.similarity_fn_name == ScoringFunction.COSINE
