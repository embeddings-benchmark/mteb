from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import mteb
from mteb.models.model_implementations import hear_s11_models


class _FakeSamples:
    def __init__(self, data, sample_rate=16000):
        self.data = data
        self.sample_rate = sample_rate


class _FakeAudioDecoder:
    def __init__(self, data, sample_rate=16000):
        self._data = data
        self.sample_rate = sample_rate
        self.sampling_rate = sample_rate

    def get_all_samples(self):
        return _FakeSamples(self._data, self.sample_rate)


class _FakeHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pooler_output_size=384)

    def forward(self, input_values, return_dict=True):  # noqa: PLR6301
        del return_dict
        pooled = input_values.float().mean(dim=1, keepdim=True).repeat(1, 384)
        return SimpleNamespace(pooler_output=pooled)


def _collate_audio(batch):
    return {"audio": [item["audio"] for item in batch]}


def test_hear_s11_model_meta_is_registered():
    meta = mteb.get_model_meta(
        "matthewagi/HeAR-s1.1",
        "a5776bebff935a81c79720467ae1e10a4effe10e",
    )

    assert meta.name == "matthewagi/HeAR-s1.1"
    assert meta.embed_dim == 384
    assert meta.modalities == ["audio"]
    assert meta.extra_requirements_groups == ["audio", "timm"]


def test_hear_s11_get_model_and_encode_support_audio_decoder(monkeypatch):
    pytest.importorskip("torchaudio")
    pytest.importorskip("timm")

    loaded = {}

    def _from_pretrained(model_ref, **kwargs):
        loaded["model_ref"] = model_ref
        loaded["kwargs"] = kwargs
        return _FakeHFModel()

    monkeypatch.setattr(hear_s11_models.AutoModel, "from_pretrained", _from_pretrained)

    model = mteb.get_model(
        "matthewagi/HeAR-s1.1",
        "a5776bebff935a81c79720467ae1e10a4effe10e",
        device="cpu",
        model_name_or_path="local-fake-hear-s11",
        amp_enabled=False,
    )
    assert loaded == {
        "model_ref": "local-fake-hear-s11",
        "kwargs": {"trust_remote_code": True},
    }
    assert model.mteb_model_meta.name == "matthewagi/HeAR-s1.1"

    rows = [
        {"audio": {"array": np.ones(32000, dtype=np.float32), "sampling_rate": 16000}},
        {"audio": _FakeAudioDecoder(torch.ones(40000), sample_rate=16000)},
    ]
    loader = DataLoader(rows, batch_size=1, shuffle=False, collate_fn=_collate_audio)

    embeddings = model.encode(
        loader,
        task_metadata=SimpleNamespace(name="BeijingOpera"),
        hf_split="test",
        hf_subset="default",
        batch_size=1,
        clip_batch_size=4,
        show_progress_bar=False,
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
