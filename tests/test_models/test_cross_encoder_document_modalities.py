"""Explicit document inputs are enforced and retained as model experiments."""

import json
from typing import Any

import numpy as np
import pytest
from datasets import Dataset, Image
from PIL import Image as PILImage

import mteb
from mteb._create_dataloaders import create_dataloader
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models import ModelMeta
from mteb.models.model_implementations.qwen3_vl_reranker import Qwen3VLRerankerWrapper
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper


@pytest.fixture
def cross_encoder_meta(monkeypatch):
    class RecordingCrossEncoder:
        def __init__(self, *args: Any, **kwargs: Any):
            assert "document_modalities" not in kwargs
            self.pairs = []

        def predict(self, pairs, **kwargs: Any):
            self.pairs = pairs
            return np.arange(len(pairs), dtype=float)

    monkeypatch.setattr("sentence_transformers.CrossEncoder", RecordingCrossEncoder)
    return ModelMeta.create_empty(
        overwrites={
            "name": "mock/multimodal-reranker",
            "revision": "1",
            "loader": CrossEncoderWrapper,
            "modalities": ["text", "image"],
        }
    )


@pytest.mark.parametrize("loader", [CrossEncoderWrapper, Qwen3VLRerankerWrapper])
def test_document_modes_survive_evaluation_cache_and_export(
    tmp_path, cross_encoder_meta, loader
):
    cross_encoder_meta = cross_encoder_meta.model_copy(update={"loader": loader})
    task = MockRetrievalTask()
    task.load_data()
    task.metadata = task.metadata.model_copy(
        update={
            "category": "t2it",
            "modalities": ["text", "image"],
            "eval_splits": ["test"],
        }
    )
    data = task.dataset["default"]["test"]
    data["corpus"] = (
        data["corpus"]
        .add_column(
            "image",
            [
                Image().encode_example(PILImage.new("RGB", (2, 2), color))
                for color in ("red", "blue")
            ],
        )
        .cast_column("image", Image())
    )
    task.dataset = {"default": {"test": data}}
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "mteb_model_meta": {"model_name": "mock/retriever", "revision": "1"},
                "default": {
                    "test": {"q1": {"d1": 2, "d2": 1}, "q2": {"d2": 2, "d1": 1}}
                },
            }
        )
    )
    task.convert_to_reranking(predictions, top_k=2)
    cache = mteb.ResultCache(tmp_path / "cache")
    paths = set()
    for modalities in (["text"], ["image"], ["text", "image"]):
        model = cross_encoder_meta.load_model(document_modalities=modalities)
        result = mteb.evaluate(model, task, cache=cache, co2_tracker=False)[0]
        assert model.mteb_model_meta.modalities == ["text", "image"]
        assert model.mteb_model_meta.experiment_kwargs == {
            "document_modalities": modalities
        }
        pairs = model.model.pairs
        assert len(pairs) == 4
        assert [q for q, _ in pairs] == [
            data["queries"]["text"][0],
            data["queries"]["text"][0],
            data["queries"]["text"][1],
            data["queries"]["text"][1],
        ]
        if modalities == ["text"]:
            assert all(isinstance(d, str) for _, d in pairs)
            assert pairs[0][1] == "Title of d2 This is another positive sentence"
        else:
            assert all(set(d) == set(modalities) for _, d in pairs)
            assert pairs[0][1]["image"].getpixel((0, 0)) == (0, 0, 255)
        path = cache.get_task_result_path(
            task.metadata.name, model.mteb_model_meta, reranking=result.reranking
        )
        paths.add(path)
        saved_meta = json.loads((path.parents[2] / "model_meta.json").read_text())
        assert saved_meta["experiment_kwargs"] == {"document_modalities": modalities}
        assert result.reranking == task.reranking_configuration
        model.model.pairs = []
        cached = mteb.evaluate(
            model, task, cache=cache, co2_tracker=False, overwrite_strategy="only-cache"
        )[0]
        assert cached.get_score() == result.get_score()
        assert model.model.pairs == []
        rows = cache.load_results(
            models=[model.mteb_model_meta], include_remote=False
        )._to_dataset()
        assert len(rows) == 1
        assert rows[0]["experiments"]["document_modalities"] == modalities
    assert len(paths) == 3


def test_text_selection_does_not_decode_excluded_images(cross_encoder_meta):
    task = MockRetrievalTask()
    task.metadata = task.metadata.model_copy(
        update={"category": "t2it", "modalities": ["text", "image"]}
    )
    # An invalid image proves the excluded column is not decoded while iterating.
    dataset = Dataset.from_dict(
        {"text": ["document"], "image": [{"bytes": b"invalid", "path": None}]}
    ).cast_column("image", Image())
    loader = create_dataloader(dataset, task_metadata=task.metadata)
    model = cross_encoder_meta.load_model(document_modalities=["text"])
    assert model._collect_inputs(loader, "prefix: ", model.document_modalities) == [
        "prefix: document"
    ]
    assert "image" in loader.dataset.features


@pytest.mark.parametrize("modalities", [[], ["unknown"]])
def test_invalid_document_modes_fail_before_loading(modalities, cross_encoder_meta):
    with pytest.raises(ValueError, match="non-empty selection"):
        cross_encoder_meta.load_model(document_modalities=modalities)


def test_missing_or_unsupported_document_modes_fail(cross_encoder_meta):
    task = MockRetrievalTask()
    loader = create_dataloader(
        Dataset.from_dict({"text": ["document"]}), task_metadata=task.metadata
    )
    model = cross_encoder_meta.load_model(document_modalities=["image"])
    with pytest.raises(ValueError, match="modalities are missing"):
        model._collect_inputs(loader, "", model.document_modalities)
    model = cross_encoder_meta.load_model(document_modalities=["audio"])
    with pytest.raises(ValueError, match="not supported"):
        model._collect_inputs(loader, "", model.document_modalities)


def test_default_and_direct_construction(cross_encoder_meta):
    model = cross_encoder_meta.load_model()
    assert model.document_modalities is None
    assert model.mteb_model_meta.experiment_kwargs is None
    direct = CrossEncoderWrapper("mock/model", document_modalities=["text"])
    assert direct.mteb_model_meta.experiment_kwargs == {"document_modalities": ["text"]}
