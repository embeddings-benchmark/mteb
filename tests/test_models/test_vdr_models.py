from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

import mteb
from mteb.models.model_implementations.vdr_models import VDRModel


class _FakeProcessor:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text=None, images=None, videos=None, padding=None, return_tensors=None):
        self.calls.append({"text": text, "images": images})
        batch_size = len(text)
        mode_value = 1.0 if text and "Query:" in text[0] else 2.0
        return {
            "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "mode": torch.full((batch_size, 1), mode_value, dtype=torch.float32),
        }


class _FakeModel:
    def __init__(self):
        self.padding_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs

    def __call__(self, **kwargs):
        mode = kwargs["mode"]
        hidden = torch.cat([mode, mode], dim=1).unsqueeze(1)
        return types.SimpleNamespace(hidden_states=[hidden, hidden])


class _FakeLoader:
    def __init__(self, batches: list[dict], features: dict[str, object]):
        self._batches = batches
        self.dataset = types.SimpleNamespace(features=features)

    def __iter__(self):
        return iter(self._batches)


class _FakeImage:
    def __init__(self, height: int = 56, width: int = 56):
        self.height = height
        self.width = width

    def resize(self, size):
        self.width, self.height = size
        return self


def _task_metadata():
    return mteb.get_task("STS12").metadata


def _mock_transformers(monkeypatch):
    fake_tf_module = types.ModuleType("transformers")
    fake_tf_module.AutoProcessor = _FakeProcessor
    fake_tf_module.Qwen2VLForConditionalGeneration = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf_module)


def test_vdr_meta_includes_image_modality():
    meta = mteb.get_model_meta("llamaindex/vdr-2b-multi-v1")
    assert meta is not None
    assert set(meta.modalities) == {"text", "image"}


def test_multimodal_wrapper_encode_text_and_image(monkeypatch):
    _mock_transformers(monkeypatch)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
    )

    text_loader = _FakeLoader(
        batches=[{"text": ["q1", "q2"]}],
        features={"text": object()},
    )
    text_embeddings = model.encode(
        text_loader,
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
    )
    assert text_embeddings.shape == (2, 2)
    assert np.allclose(text_embeddings, 1.0)

    image_loader = _FakeLoader(
        batches=[{"image": [_FakeImage(), _FakeImage()]}],
        features={"image": object()},
    )
    image_embeddings = model.encode(
        image_loader,
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
    )
    assert image_embeddings.shape == (2, 2)
    assert np.allclose(image_embeddings, 2.0)

    mixed_loader = _FakeLoader(
        batches=[
            {
                "text": ["q1", "q2"],
                "image": [_FakeImage(), _FakeImage()],
            }
        ],
        features={"text": object(), "image": object()},
    )
    mixed_embeddings = model.encode(
        mixed_loader,
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
    )
    assert mixed_embeddings.shape == (2, 2)
    assert np.allclose(mixed_embeddings, 3.0)


def test_multimodal_wrapper_raises_on_mismatched_text_image_lengths(monkeypatch):
    _mock_transformers(monkeypatch)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
    )

    with pytest.raises(ValueError, match="same length"):
        model.encode(
            _FakeLoader(
                batches=[{"text": ["q1"], "image": [_FakeImage(), _FakeImage()]}],
                features={"text": object(), "image": object()},
            ),
            task_metadata=_task_metadata(),
            hf_split="test",
            hf_subset="default",
        )


def test_multimodal_wrapper_raises_on_unsupported_empty_features(monkeypatch):
    _mock_transformers(monkeypatch)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
    )

    with pytest.raises(ValueError, match="No text or image features"):
        model.encode(
            _FakeLoader(batches=[{}], features={}),
            task_metadata=_task_metadata(),
            hf_split="test",
            hf_subset="default",
        )


def test_multimodal_wrapper_outputs_numpy(monkeypatch):
    _mock_transformers(monkeypatch)
    model = VDRModel(
        model_name="llamaindex/vdr-2b-multi-v1",
        revision="dummy",
    )

    emb = model.encode(
        _FakeLoader(
            batches=[{"text": ["q1", "q2"]}],
            features={"text": object()},
        ),
        task_metadata=_task_metadata(),
        hf_split="test",
        hf_subset="default",
    )
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 2)
