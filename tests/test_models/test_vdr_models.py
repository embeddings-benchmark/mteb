from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

import mteb
from mteb.models.model_implementations.vdr_models import VDRModel
from mteb.types import PromptType


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.max_seq_length = kwargs.get("max_seq_length", None)

    def set_pooling_include_prompt(self, include_prompt: bool):
        return None

    def encode(self, items, **kwargs):
        self.calls.append((items, kwargs))
        n = len(items)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if isinstance(items[0], str):
            return np.ones((n, 2), dtype=np.float32)
        return np.full((n, 2), 2.0, dtype=np.float32)


class _TensorSentenceTransformer(_FakeSentenceTransformer):
    def encode(self, items, **kwargs):
        arr = super().encode(items, **kwargs)
        return torch.tensor(arr)


class _FakeLoader:
    def __init__(self, batches: list[dict], features: dict[str, object]):
        self._batches = batches
        self.dataset = types.SimpleNamespace(features=features)

    def __iter__(self):
        return iter(self._batches)


def _task_metadata():
    return mteb.get_task("STS12").metadata


def test_vdr_meta_includes_image_modality():
    meta = mteb.get_model_meta("llamaindex/vdr-2b-multi-v1")
    assert meta is not None
    assert set(meta.modalities) == {"text", "image"}


def test_vdr_encode_text_and_image(monkeypatch):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
        instruction_template="{instruction}",
    )
    task_metadata = _task_metadata()

    text_loader = _FakeLoader(
        batches=[{"text": ["q1", "q2"]}],
        features={"text": object()},
    )
    text_embeddings = model.encode(
        text_loader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )
    assert text_embeddings.shape == (2, 2)
    assert np.allclose(text_embeddings, 1.0)
    _, text_kwargs = model.model.calls[-1]
    assert "prompt" in text_kwargs

    image_loader = _FakeLoader(
        batches=[{"image": [object(), object()]}],
        features={"image": object()},
    )
    image_embeddings = model.encode(
        image_loader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.document,
    )
    assert image_embeddings.shape == (2, 2)
    assert np.allclose(image_embeddings, 2.0)

    mixed_loader = _FakeLoader(
        batches=[
            {
                "text": ["q1", "q2"],
                "image": [object(), object()],
            }
        ],
        features={"text": object(), "image": object()},
    )
    mixed_embeddings = model.encode(
        mixed_loader,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )
    assert mixed_embeddings.shape == (2, 2)
    assert np.allclose(mixed_embeddings, 3.0)


def test_vdr_raises_on_mismatched_text_image_lengths(monkeypatch):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
        instruction_template="{instruction}",
    )

    with pytest.raises(ValueError, match="same length"):
        model.encode(
            _FakeLoader(
                batches=[{"text": ["q1"], "image": [object(), object()]}],
                features={"text": object(), "image": object()},
            ),
            task_metadata=_task_metadata(),
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
        )


def test_vdr_raises_on_unsupported_empty_features(monkeypatch):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
        instruction_template="{instruction}",
    )

    with pytest.raises(ValueError, match="No text or image features"):
        model.encode(
            _FakeLoader(batches=[{}], features={}),
            task_metadata=_task_metadata(),
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
        )


def test_vdr_converts_tensor_outputs_to_numpy(monkeypatch):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _TensorSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
        instruction_template="{instruction}",
    )

    emb = model.encode(
        _FakeLoader(
            batches=[{"text": ["q1", "q2"]}],
            features={"text": object()},
        ),
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 2)


def test_vdr_no_prompt_on_document_when_disabled(monkeypatch):
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
        instruction_template="{instruction}",
        apply_instruction_to_passages=False,
    )

    model.encode(
        _FakeLoader(
            batches=[{"text": ["doc1"]}],
            features={"text": object()},
        ),
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.document,
    )
    _, kwargs = model.model.calls[-1]
    assert kwargs.get("prompt") is None
