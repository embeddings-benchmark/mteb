from __future__ import annotations

from typing import Any

from scripts import run_moving_fashion_v2i_pair_baseline as runner


def test_loads_ebind_directly_with_pinned_revision(monkeypatch) -> None:
    received: dict[str, Any] = {}
    expected_model = object()

    def fake_ebind_wrapper(**kwargs):
        received.update(kwargs)
        return expected_model

    monkeypatch.setattr(runner, "EBindWrapper", fake_ebind_wrapper)

    model = runner._load_model(
        runner._DEFAULT_MODEL,
        "cuda",
        {"fps": None, "num_frames": 8},
    )

    assert model is expected_model
    assert received == {
        "model_name": "encord-team/ebind-audio-vision",
        "revision": runner.ebind_audio_vision.revision,
        "device": "cuda",
        "fps": None,
        "num_frames": 8,
    }


def test_loads_other_models_through_mteb_registry(monkeypatch) -> None:
    received: dict[str, Any] = {}
    expected_model = object()

    def fake_get_model(model_name, **kwargs):
        received["model_name"] = model_name
        received.update(kwargs)
        return expected_model

    monkeypatch.setattr(runner.mteb, "get_model", fake_get_model)

    model = runner._load_model("another/model", "cuda", {"num_frames": 4})

    assert model is expected_model
    assert received == {
        "model_name": "another/model",
        "device": "cuda",
        "num_frames": 4,
    }
