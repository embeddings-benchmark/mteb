"""Document selection must control model inputs and result-cache identity together."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Image
from PIL import Image as PILImage

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask
from mteb.models import (
    CachedEmbeddingWrapper,
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
)
from mteb.models.model_meta import ScoringFunction
from mteb.models.search_encoder_index.search_indexes.faiss_search_index import (
    FaissSearchIndex,
)
from mteb.types import Modalities, PromptType


class RecordingReranker:
    def __init__(self):
        self.mteb_model_meta = mteb.get_model_meta(
            "mteb/baseline-random-cross-encoder"
        ).model_copy(
            update={
                "modalities": ["text", "image"],
                "experiment_kwargs": {"existing_option": True},
            },
            deep=True,
        )
        self.calls = []

    def predict(self, inputs1, inputs2, **kwargs: Any):
        queries, documents = list(inputs1), list(inputs2)
        self.calls.append((queries, documents, kwargs))
        # Return one score per pair, preferring d1 for text and d2 for images.
        return np.array(
            [
                float(doc_id == ("d1" if "text" in batch else "d2"))
                for batch in documents
                for doc_id in batch["id"]
            ]
        )


@pytest.fixture
def multimodal_task():
    task = MockRerankingTask()
    task.metadata = task.metadata.model_copy(
        update={"modalities": ["text", "image"], "category": "it2it"}
    )
    task.load_data()
    split = task.dataset["default"]["test"]
    for key in ("corpus", "queries"):
        split[key] = (
            split[key]
            .add_column(
                "image",
                [
                    Image().encode_example(PILImage.new("RGB", (2, 2), (i * 100, 0, 0)))
                    for i in range(len(split[key]))
                ],
            )
            .cast_column("image", Image())
        )
    return task


@pytest.mark.parametrize("modalities", [None, ["text"], ["image"], ["text", "image"]])
def test_selected_document_inputs_leave_queries_and_task_unchanged(
    modalities: list[Modalities] | None, multimodal_task
):
    model = RecordingReranker()
    wrapper = SearchCrossEncoderWrapper(model, document_modalities=modalities)
    original_meta = model.mteb_model_meta.model_dump()
    mteb.evaluate(wrapper, multimodal_task, cache=None, co2_tracker=False)
    queries, documents, kwargs = model.calls[0]
    assert all({"text", "image", "query"}.issubset(batch) for batch in queries)
    expected = set(modalities or ["text", "image"])
    assert all(set(batch) & {"text", "image"} == expected for batch in documents)
    assert all(("title" in batch) == ("text" in expected) for batch in documents)
    assert all(("body" in batch) == ("text" in expected) for batch in documents)
    assert [doc_id for batch in documents for doc_id in batch["id"]] == [
        "d1",
        "d2",
        "d2",
        "d1",
    ]
    assert kwargs["task_metadata"] is multimodal_task.metadata
    assert multimodal_task.metadata.category == "it2it"
    assert "image" in multimodal_task.dataset["default"]["test"]["corpus"].column_names
    assert model.mteb_model_meta.model_dump() == original_meta


@pytest.mark.parametrize("encoder", [False, True])
def test_document_modalities_cache_round_trip(tmp_path: Path, multimodal_task, encoder):
    model = RecordingEncoder() if encoder else RecordingReranker()
    wrapper_cls = SearchEncoderWrapper if encoder else SearchCrossEncoderWrapper
    cache = mteb.ResultCache(tmp_path)
    paths = set()
    for modalities in (None, ["text"], ["image"], ["text", "image"]):
        wrapper = wrapper_cls(model, document_modalities=modalities)
        result = mteb.evaluate(wrapper, multimodal_task, cache=cache, co2_tracker=False)
        path = cache.get_task_result_path(
            multimodal_task.metadata.name, wrapper.mteb_model_meta
        )
        paths.add(path)
        saved_meta = json.loads((path.parent / "model_meta.json").read_text())
        expected_kwargs = {"existing_option": True}
        if modalities:
            expected_kwargs["document_modalities"] = sorted(modalities)
        assert saved_meta["experiment_kwargs"] == expected_kwargs
        cached = cache.load_task_result(
            multimodal_task.metadata.name, wrapper.mteb_model_meta
        )
        assert cached.get_score() == result[0].get_score()
    assert len(paths) == 4
    calls = len(model.calls)
    # Reordered/duplicated selections reuse the same complete cache entry.
    reordered = wrapper_cls(model, document_modalities=["image", "text", "image"])
    mteb.evaluate(reordered, multimodal_task, cache=cache, co2_tracker=False)
    assert len(model.calls) == calls
    loaded = list(
        cache.load_results(include_remote=False, load_experiments="match_name")
    )
    assert len(loaded) == 4
    assert {
        tuple((r.model_meta.experiment_kwargs or {}).get("document_modalities", []))
        for r in loaded
    } == {(), ("text",), ("image",), ("image", "text")}
    assert model.mteb_model_meta.experiment_kwargs == {"existing_option": True}


@pytest.mark.parametrize("modalities", [[], "text", ["invalid"]])
def test_reject_invalid_document_modalities(modalities):
    with pytest.raises(ValueError, match="non-empty sequence"):
        SearchCrossEncoderWrapper(RecordingReranker(), document_modalities=modalities)


def test_reject_unsupported_and_missing_document_modalities(multimodal_task):
    model = RecordingReranker()
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"modalities": ["text"]}
    )
    with pytest.raises(ValueError, match="Model supports"):
        SearchCrossEncoderWrapper(model, document_modalities=["image"])
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"modalities": ["text", "image"]}
    )
    wrapper = SearchCrossEncoderWrapper(model, document_modalities=["image"])
    with pytest.raises(ValueError, match="missing document modalities.*image"):
        mteb.evaluate(wrapper, MockRerankingTask(), cache=None, co2_tracker=False)
    split = multimodal_task.dataset["default"]["test"]
    split["corpus"] = split["corpus"].remove_columns("image")
    with pytest.raises(ValueError, match="missing document modalities.*image"):
        mteb.evaluate(wrapper, multimodal_task, cache=None, co2_tracker=False)
    assert not model.calls


def test_text_selection_does_not_decode_document_images(multimodal_task):
    split = multimodal_task.dataset["default"]["test"]
    split["corpus"] = (
        split["corpus"]
        .remove_columns("image")
        .add_column("image", ["/nonexistent/document.png"] * len(split["corpus"]))
        .cast_column("image", Image())
    )
    wrapper = SearchCrossEncoderWrapper(
        RecordingReranker(), document_modalities=["text"]
    )
    mteb.evaluate(wrapper, multimodal_task, cache=None, co2_tracker=False)


def test_load_model_restores_document_selection(multimodal_task):
    received_kwargs = []

    def loader(name, revision, **kwargs: Any):
        received_kwargs.append(kwargs)
        return RecordingReranker()

    meta = RecordingReranker().mteb_model_meta.model_copy(update={"loader": loader})
    model = meta.load_model(document_modalities=["image"])
    restored = model.mteb_model_meta.load_model()
    mteb.evaluate(restored, multimodal_task, cache=None, co2_tracker=False)
    assert all("document_modalities" not in kwargs for kwargs in received_kwargs)
    assert (
        model.mteb_model_meta.experiment_name
        == restored.mteb_model_meta.experiment_name
    )
    assert restored.document_modalities == ("image",)
    assert all("text" not in batch for batch in restored.model.calls[0][1])
    assert "document_modalities" not in meta.experiment_kwargs


class RecordingEncoder:
    def __init__(self):
        self.mteb_model_meta = RecordingReranker().mteb_model_meta.model_copy(
            update={"similarity_fn_name": ScoringFunction.DOT_PRODUCT}
        )
        self.calls = []

    def encode(self, inputs, *, prompt_type=None, **kwargs: Any):
        batches = list(inputs)
        self.calls.append((prompt_type, batches))
        return np.array(
            [
                [1.0, 0.0]
                if prompt_type == PromptType.query
                else [
                    float("text" in batch) + 2 * float("image" in batch),
                    float(doc_id[-1]),
                ]
                for batch in batches
                for doc_id in batch["id"]
            ],
            dtype=np.float32,
        )

    def similarity(self, a, b):
        return a @ b.T

    def similarity_pairwise(self, a, b):
        return (a * b).sum(axis=-1)


@pytest.mark.parametrize("modalities", [["text"], ["image"], ["text", "image"]])
@pytest.mark.parametrize("route", ["full", "candidates", "index"])
def test_encoder_document_selection(modalities, route, multimodal_task):
    model = RecordingEncoder()
    if route != "candidates":
        multimodal_task.dataset["default"]["test"]["top_ranked"] = None
    if route == "index":
        pytest.importorskip("faiss")
    index = FaissSearchIndex(model) if route == "index" else None
    wrapper = SearchEncoderWrapper(
        model, document_modalities=modalities, index_backend=index
    )
    mteb.evaluate(wrapper, multimodal_task, cache=None, co2_tracker=False)
    for prompt_type, batches in model.calls:
        expected = (
            set(modalities) if prompt_type == PromptType.document else {"text", "image"}
        )
        assert all(set(batch) & {"text", "image"} == expected for batch in batches)
        if prompt_type == PromptType.document:
            assert all(("title" in batch) == ("text" in expected) for batch in batches)
            assert all(("body" in batch) == ("text" in expected) for batch in batches)
    assert model.mteb_model_meta.experiment_kwargs == {"existing_option": True}
    assert multimodal_task.metadata.category == "it2it"
    assert "image" in multimodal_task.dataset["default"]["test"]["corpus"].column_names


def test_encoder_embedding_cache_preserves_document_selection(
    tmp_path: Path, multimodal_task
):
    split = multimodal_task.dataset["default"]["test"]
    model = RecordingEncoder()
    cached = CachedEmbeddingWrapper(model, tmp_path)
    selections = [["text"], ["image"], ["text", "image"]]

    def search(modalities):
        wrapper = SearchEncoderWrapper(cached, document_modalities=modalities)
        kwargs = dict(
            task_metadata=multimodal_task.metadata,
            hf_subset="default",
            hf_split="test",
            encode_kwargs={},
        )
        wrapper.index(split["corpus"], **kwargs)
        return wrapper.search(split["queries"], top_k=2, **kwargs)

    results = {}
    for modalities in selections:
        result = search(modalities)
        expected_score = float("text" in modalities) + 2 * float("image" in modalities)
        assert all(
            score == expected_score
            for documents in result.values()
            for score in documents.values()
        )
        results[tuple(modalities)] = result
        mode, batches = model.calls[-1]
        assert mode == PromptType.document
        assert all(
            set(batch) & {"text", "image"} == set(modalities) for batch in batches
        )
    calls = len(model.calls)
    assert calls == 4  # Shared query embeddings, three distinct document inputs.
    # Repeat in reverse order: all inputs must reuse their own embeddings.
    for modalities in reversed(selections):
        assert search(modalities) == results[tuple(modalities)]
    assert len(model.calls) == calls
    doc_cache = next(
        cache
        for (_, prompt_type, _), cache in cached.cache_dict.items()
        if prompt_type == PromptType.document
    )
    assert len(doc_cache.hash_to_index) == 6  # Two documents in each representation.


def test_get_model_document_selection_for_encoders(multimodal_task):
    model = mteb.get_model(
        "mteb/baseline-random-encoder", document_modalities=["image"]
    )
    assert isinstance(model, SearchEncoderWrapper)
    assert model.document_modalities == ("image",)
    restored = model.mteb_model_meta.load_model()
    assert isinstance(restored, SearchEncoderWrapper)
    assert restored.document_modalities == ("image",)
    assert (
        restored.mteb_model_meta.experiment_name
        == model.mteb_model_meta.experiment_name
    )
    mteb.evaluate(restored, multimodal_task, cache=None, co2_tracker=False)


def test_get_model_rejects_custom_search_document_selection():
    class CustomSearchEncoder(RecordingEncoder):
        def index(self, *args: Any, **kwargs: Any):
            pass

        def search(self, *args: Any, **kwargs: Any):
            pass

    meta = RecordingEncoder().mteb_model_meta.model_copy(
        update={"loader": lambda name, revision, **kwargs: CustomSearchEncoder()}
    )
    with pytest.raises(ValueError, match="custom search implementation"):
        meta.load_model(document_modalities=["text"])
