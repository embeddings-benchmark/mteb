from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import Dataset

from scripts import run_moving_fashion_v2i_pair_baseline as runner


def test_loads_ebind_directly_with_pinned_revision(monkeypatch) -> None:
    received: dict[str, Any] = {}
    expected_model = SimpleNamespace()

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
    assert model.mteb_model_meta.modalities == ["text", "image", "audio", "video"]
    assert model.mteb_model_meta.experiment_kwargs == {
        "fps": None,
        "num_frames": 8,
    }
    assert model.mteb_model_meta.loader_kwargs == {
        "device": "cuda",
        "fps": None,
        "num_frames": 8,
    }
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


def test_builds_model_and_experiment_specific_prediction_folder() -> None:
    model = SimpleNamespace(
        mteb_model_meta=SimpleNamespace(
            name="organization/model",
            revision="revision-1",
            experiment_kwargs={"num_frames": 8, "fps": None},
        )
    )

    folder = runner._default_prediction_folder(Path("results"), model)

    assert folder == Path(
        "results/predictions/organization__model/revision-1/fps_None__num_frames_8"
    )


def test_writes_huggingface_columns_as_json_lists(tmp_path) -> None:
    task = runner.MovingFashionV2IPairClassification()
    task.dataset = {
        "test": Dataset.from_dict(
            {
                "video_id": ["v1", "v1"],
                "image_id": ["i1", "i2"],
                "label": [1, 0],
                "source_subset": ["hard", "hard"],
            }
        )
    }
    task.data_loaded = True
    output_path = tmp_path / "pairs.json"

    runner._write_pair_manifest(task, output_path)

    manifest = json.loads(output_path.read_text())
    assert manifest["rows"] == {
        "video_id": ["v1", "v1"],
        "image_id": ["i1", "i2"],
        "label": [1, 0],
        "source_subset": ["hard", "hard"],
    }


def test_finds_single_saved_prediction_folder(tmp_path) -> None:
    expected_folder = tmp_path / "predictions" / "model" / "experiment"
    expected_folder.mkdir(parents=True)
    (expected_folder / "Task_predictions.json").write_text("{}")

    assert (
        runner._find_prediction_folder(tmp_path, None, "Task_predictions.json")
        == expected_folder
    )
