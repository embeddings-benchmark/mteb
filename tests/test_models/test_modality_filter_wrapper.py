"""Behavioral tests for modality selection in retrieval and reranking."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask, MockRetrievalTask
from mteb.models import ModalityFilterWrapper, ModelMeta, SearchEncoderWrapper
from mteb.models.cache_wrappers import CachedEmbeddingWrapper
from mteb.models.model_meta import ScoringFunction
from mteb.models.models_protocols import (
    CrossEncoderProtocol,
    EncoderProtocol,
    SearchProtocol,
)
from mteb.types import PromptType


def _meta(**updates: Any):
    return ModelMeta.create_empty(
        overwrites={
            # Synthetic baselines do not need optional vision-model dependencies.
            "name": "mteb/baseline-modality-selection",
            "revision": "test-revision",
            "modalities": ["text", "image"],
            "similarity_fn_name": ScoringFunction.DOT_PRODUCT,
            **updates,
        }
    )


class RecordingSearch:
    def __init__(self, **meta_updates: Any):
        self.mteb_model_meta = _meta(**meta_updates)
        self.index_calls = []
        self.search_calls = []

    def index(self, corpus, **kwargs: Any):
        self.index_calls.append((corpus, kwargs))
        self.corpus = corpus

    def search(self, queries, **kwargs: Any):
        self.search_calls.append((queries, kwargs))
        candidates = kwargs.get("top_ranked")
        return {
            qid: {
                docid: float(i + 1)
                for i, docid in enumerate(
                    candidates[qid] if candidates else self.corpus["id"]
                )
            }
            for qid in queries["id"]
        }


class RecordingEncoder:
    def __init__(self):
        self.mteb_model_meta = _meta()
        self.calls = []

    def encode(self, inputs, *, task_metadata, prompt_type=None, **kwargs: Any):
        batches = list(inputs)
        self.calls.append((prompt_type, task_metadata, batches))
        # Deliberately depend on corpus modality even for query embeddings.
        value = 2.0 if task_metadata.category.endswith("i") else 1.0
        return np.full((len(inputs.dataset), 2), value, dtype=np.float32)

    def similarity(self, embeddings1, embeddings2):
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        return (embeddings1 * embeddings2).sum(axis=-1)


class RecordingCrossEncoder:
    def __init__(self):
        self.mteb_model_meta = _meta(model_type=["cross-encoder"])
        self.calls = []

    def predict(self, inputs1, inputs2, *, task_metadata, **kwargs: Any):
        self.calls.append((list(inputs1), list(inputs2), task_metadata, kwargs))
        return np.arange(len(inputs2.dataset), dtype=np.float32)


@pytest.fixture
def multimodal_inputs():
    corpus = Dataset.from_dict(
        {
            "id": ["doc1", "doc2"],
            "text": ["body one", "body two"],
            "title": ["title one", "title two"],
            "body": ["body one", "body two"],
            "image": [PILImage.new("RGB", (2, 2), color) for color in ("red", "blue")],
        },
        features=Features(
            {
                "id": Value("string"),
                "text": Value("string"),
                "title": Value("string"),
                "body": Value("string"),
                "image": Image(),
            }
        ),
    )
    queries = Dataset.from_dict(
        {"id": ["q1"], "text": ["question"], "instruction": ["find the page"]}
    )
    metadata = MockRetrievalTask.metadata.model_copy(
        update={"category": "t2it", "modalities": ["text", "image"]}, deep=True
    )
    return corpus, queries, metadata


def _run_search(model, inputs, **search_kwargs: Any):
    corpus, queries, metadata = inputs
    common = dict(
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        encode_kwargs={"batch_size": 2},
        num_proc=None,
    )
    model.index(corpus, **common)
    return model.search(queries, top_k=2, **common, **search_kwargs)


@pytest.mark.parametrize("selection", [["text"], ["image"], ["text", "image"]])
def test_corpus_selection_preserves_queries_and_search_options(
    multimodal_inputs, selection
):
    base = RecordingSearch(experiment_kwargs={"custom_setting": 7})
    wrapper = ModalityFilterWrapper(base, corpus_modalities=selection)
    corpus, queries, metadata = multimodal_inputs
    candidates = {"q1": ["doc2"]}

    result = _run_search(
        wrapper, multimodal_inputs, top_ranked=candidates, custom_option=17
    )

    filtered, index_kwargs = base.index_calls[0]
    forwarded_queries, search_kwargs = base.search_calls[0]
    assert set(filtered.column_names) & {"text", "image"} == set(selection)
    if "text" not in selection:
        assert not {"text", "title", "body"} & set(filtered.column_names)
    assert filtered["id"] == corpus["id"]
    assert forwarded_queries.to_dict() == queries.to_dict()
    assert search_kwargs["top_ranked"] == candidates
    assert search_kwargs["custom_option"] == 17
    assert search_kwargs["top_k"] == 2
    assert index_kwargs["hf_split"] == "test"
    assert index_kwargs["hf_subset"] == "default"
    for kwargs in (index_kwargs, search_kwargs):
        effective = kwargs["task_metadata"]
        assert set(effective.get_modalities(PromptType.document)) == set(selection)
        assert effective.get_modalities(PromptType.query) == ["text"]
        assert effective.name == metadata.name
        assert effective.dataset == metadata.dataset
    assert result == {"q1": {"doc2": 1.0}}
    assert metadata.category == "t2it"
    assert set(corpus.column_names) == {"id", "text", "title", "body", "image"}
    assert base.mteb_model_meta.experiment_kwargs == {"custom_setting": 7}
    assert base.mteb_model_meta.modalities == ["text", "image"]
    assert wrapper.mteb_model_meta.experiment_kwargs["custom_setting"] == 7
    assert set(wrapper.mteb_model_meta.experiment_kwargs["corpus_modalities"]) == set(
        selection
    )


def test_query_selection_does_not_change_corpus(multimodal_inputs):
    corpus, queries, metadata = multimodal_inputs
    queries = corpus.select([0]).rename_column("id", "old_id").add_column("id", ["q1"])
    metadata = metadata.model_copy(update={"category": "it2it"})
    base = RecordingSearch()
    _run_search(
        ModalityFilterWrapper(base, query_modalities=["image"]),
        (corpus, queries, metadata),
    )
    forwarded_corpus, _ = base.index_calls[0]
    filtered_queries, kwargs = base.search_calls[0]
    assert set(forwarded_corpus.column_names) == set(corpus.column_names)
    assert not {"text", "title", "body"} & set(filtered_queries.column_names)
    assert filtered_queries["id"] == ["q1"]
    assert kwargs["task_metadata"].get_modalities(PromptType.query) == ["image"]
    assert set(kwargs["task_metadata"].get_modalities(PromptType.document)) == {
        "text",
        "image",
    }


def test_omitted_selection_preserves_existing_identity_and_inputs(multimodal_inputs):
    base = RecordingSearch(experiment_kwargs={"setting": 1})
    wrapper = ModalityFilterWrapper(base)
    _run_search(wrapper, multimodal_inputs)
    assert (
        wrapper.mteb_model_meta.experiment_name == base.mteb_model_meta.experiment_name
    )
    assert base.index_calls[0][0].column_names == multimodal_inputs[0].column_names
    assert base.search_calls[0][0].to_dict() == multimodal_inputs[1].to_dict()


def test_selection_identity_is_normalized_and_does_not_mutate_siblings():
    base = RecordingSearch()
    first = ModalityFilterWrapper(base, corpus_modalities=["text", "image", "text"])
    equivalent = ModalityFilterWrapper(base, corpus_modalities=["image", "text"])
    different = ModalityFilterWrapper(base, corpus_modalities=["image"])
    assert (
        first.mteb_model_meta.experiment_name
        == equivalent.mteb_model_meta.experiment_name
    )
    assert (
        first.mteb_model_meta.experiment_name
        != different.mteb_model_meta.experiment_name
    )
    assert base.mteb_model_meta.experiment_kwargs is None


@pytest.mark.parametrize("selection", [[], ["invalid"], "image"])
@pytest.mark.parametrize("side", ["query_modalities", "corpus_modalities"])
def test_invalid_selection_is_rejected(selection, side):
    with pytest.raises((ValueError, TypeError)):
        ModalityFilterWrapper(RecordingSearch(), **{side: selection})


def test_known_unsupported_modality_is_rejected():
    with pytest.raises(ValueError, match="image"):
        ModalityFilterWrapper(
            RecordingSearch(modalities=["text"]), corpus_modalities=["image"]
        )


def test_missing_corpus_modality_fails_before_index(multimodal_inputs):
    corpus, queries, metadata = multimodal_inputs
    base = RecordingSearch()
    wrapper = ModalityFilterWrapper(base, corpus_modalities=["text", "image"])
    with pytest.raises(ValueError, match="image"):
        _run_search(wrapper, (corpus.remove_columns("image"), queries, metadata))
    assert not base.index_calls


def test_missing_query_modality_fails_before_search(multimodal_inputs):
    base = RecordingSearch()
    wrapper = ModalityFilterWrapper(base, query_modalities=["image"])
    with pytest.raises(ValueError, match="image"):
        _run_search(wrapper, multimodal_inputs)
    assert not base.search_calls


def test_unknown_capability_does_not_prevent_valid_input_selection(multimodal_inputs):
    base = RecordingSearch(modalities=None)
    _run_search(
        ModalityFilterWrapper(base, corpus_modalities=["image"]), multimodal_inputs
    )
    assert set(base.index_calls[0][0].column_names) == {"id", "image"}


def test_custom_search_with_multiple_protocols_keeps_custom_dispatch(multimodal_inputs):
    class MultipleProtocols(RecordingSearch, RecordingEncoder, RecordingCrossEncoder):
        def encode(self, *args: Any, **kwargs: Any):
            pytest.fail(
                "custom search must not be replaced with generic encoder search"
            )

        def predict(self, *args: Any, **kwargs: Any):
            pytest.fail("custom search must not be replaced with generic reranking")

    base = MultipleProtocols()
    wrapper = ModalityFilterWrapper(base, corpus_modalities=["image"])
    assert isinstance(wrapper, SearchProtocol)
    assert not isinstance(wrapper, EncoderProtocol)
    assert not isinstance(wrapper, CrossEncoderProtocol)
    _run_search(wrapper, multimodal_inputs)
    assert len(base.index_calls) == len(base.search_calls) == 1


def test_encoder_adapter_filters_before_preparation(multimodal_inputs):
    base = RecordingEncoder()
    result = _run_search(
        ModalityFilterWrapper(base, corpus_modalities=["image"]), multimodal_inputs
    )
    assert set(result["q1"]) == {"doc1", "doc2"}
    assert {call[0] for call in base.calls} == {PromptType.query, PromptType.document}
    for role, metadata, batches in base.calls:
        assert metadata.category == "t2i"
        for batch in batches:
            assert (set(batch) & {"text", "image"}) == (
                {"text"} if role == PromptType.query else {"image"}
            )


def test_cross_encoder_adapter_preserves_candidate_pairs(multimodal_inputs):
    base = RecordingCrossEncoder()
    result = _run_search(
        ModalityFilterWrapper(base, corpus_modalities=["image"]),
        multimodal_inputs,
        top_ranked={"q1": ["doc2"]},
    )
    assert set(result["q1"]) == {"doc2"}
    query_batches, corpus_batches, metadata, kwargs = base.calls[0]
    assert query_batches[0]["text"] == ["question find the page"]
    assert set(corpus_batches[0]) & {"text", "image", "title", "body"} == {"image"}
    assert metadata.category == "t2i"
    assert kwargs["batch_size"] == 2


def test_saved_experiment_reloads_filter_without_forwarding_loader_options(
    multimodal_inputs,
):
    seen = []

    def loader(model_name, revision=None, **kwargs: Any):
        seen.append(kwargs)
        return RecordingSearch()

    base = RecordingSearch(loader=loader, experiment_kwargs={"custom_setting": 7})
    configured = ModalityFilterWrapper(
        base, corpus_modalities=["image"], query_modalities=["text"]
    )
    restored = configured.mteb_model_meta.load_model()
    _run_search(restored, multimodal_inputs)
    assert "corpus_modalities" not in seen[0]
    assert "query_modalities" not in seen[0]
    assert seen[0]["custom_setting"] == 7
    assert isinstance(restored, ModalityFilterWrapper)
    assert (
        restored.mteb_model_meta.experiment_name
        == configured.mteb_model_meta.experiment_name
    )


def test_result_directories_keep_each_modality_experiment(tmp_path):
    cache = mteb.ResultCache(cache_path=tmp_path)
    task = MockRetrievalTask()
    base = RecordingSearch()
    configured = ModalityFilterWrapper(base, corpus_modalities=["text"])
    for model in (base, configured):
        mteb.evaluate(model, task, cache=cache, co2_tracker=False)
    paths = list(tmp_path.rglob("model_meta.json"))
    assert len(paths) == 2
    saved = [json.loads(path.read_text()) for path in paths]
    assert any(meta["experiment_kwargs"] is None for meta in saved)
    assert any(
        meta["experiment_kwargs"] == {"corpus_modalities": ["text"]} for meta in saved
    )
    assert sum("experiments" in path.parts for path in paths) == 1
    result_paths = list(tmp_path.rglob("MockRetrievalTask.json"))
    assert len(result_paths) == 2


@pytest.mark.parametrize("custom_search", [False, True])
def test_inner_embedding_cache_isolates_query_and_corpus_modes(
    multimodal_inputs, tmp_path, custom_search
):
    encoder = RecordingEncoder()
    cached = CachedEmbeddingWrapper(encoder, tmp_path)
    if custom_search:

        class DelegatingSearch(RecordingSearch):
            def __init__(self):
                super().__init__()
                self.search_impl = SearchEncoderWrapper(cached)

            def index(self, corpus, **kwargs: Any):
                self.search_impl.index(corpus, **kwargs)

            def search(self, queries, **kwargs: Any):
                return self.search_impl.search(queries, **kwargs)

        base = DelegatingSearch()
    else:
        base = cached
    for selection in (["text"], ["image"], ["text"]):
        _run_search(
            ModalityFilterWrapper(base, corpus_modalities=selection), multimodal_inputs
        )
    # Third run is reused, but image-mode queries cannot reuse text-mode vectors.
    assert len(encoder.calls) == 4
    query_calls = [call for call in encoder.calls if call[0] == PromptType.query]
    assert {call[1].category for call in query_calls} == {"t2t", "t2i"}
    assert len(list(tmp_path.rglob("index.json"))) == 4
    assert cached.mteb_model_meta.experiment_kwargs is None
    assert encoder.mteb_model_meta.experiment_kwargs is None
    # After the wrapper returns, unconfigured calls use the original namespace.
    _run_search(SearchEncoderWrapper(cached), multimodal_inputs)
    assert len(encoder.calls) == 6


def test_unsupported_outer_embedding_cache_and_nested_filters_are_rejected(tmp_path):
    wrapper = ModalityFilterWrapper(RecordingEncoder(), corpus_modalities=["text"])
    with pytest.raises((ValueError, TypeError)):
        CachedEmbeddingWrapper(wrapper, tmp_path)
    with pytest.raises((ValueError, TypeError)):
        ModalityFilterWrapper(wrapper, corpus_modalities=["image"])


def test_stateful_index_backend_without_reset_is_rejected():
    class AppendOnlyIndex:
        def add_documents(self, embeddings, document_ids):
            pytest.fail("incompatible reusable index must not receive documents")

    indexed = SearchEncoderWrapper(RecordingEncoder(), index_backend=AppendOnlyIndex())
    with pytest.raises(ValueError, match="index"):
        ModalityFilterWrapper(indexed, corpus_modalities=["image"])


@pytest.mark.parametrize("failure_stage", ["index", "search"])
def test_failed_delegate_does_not_leak_cache_configuration(
    multimodal_inputs, tmp_path, failure_stage
):
    from mteb._create_dataloaders import create_dataloader

    encoder = RecordingEncoder()
    cached = CachedEmbeddingWrapper(encoder, tmp_path)

    class FailingSearch(RecordingSearch):
        def __init__(self):
            super().__init__()
            self.impl = SearchEncoderWrapper(cached)

        def index(self, corpus, **kwargs: Any):
            if failure_stage == "index":
                loader = create_dataloader(
                    corpus,
                    task_metadata=kwargs["task_metadata"],
                    prompt_type=PromptType.document,
                )
                cached.encode(
                    loader,
                    task_metadata=kwargs["task_metadata"],
                    hf_split=kwargs["hf_split"],
                    hf_subset=kwargs["hf_subset"],
                    prompt_type=PromptType.document,
                )
                raise RuntimeError("index failed after encoding")
            self.impl.index(corpus, **kwargs)

        def search(self, queries, **kwargs: Any):
            self.impl.search(queries, **kwargs)
            raise RuntimeError("search failed after encoding")

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        _run_search(
            ModalityFilterWrapper(FailingSearch(), corpus_modalities=["image"]),
            multimodal_inputs,
        )
    calls_before = len(encoder.calls)
    _run_search(SearchEncoderWrapper(cached), multimodal_inputs)
    assert len(encoder.calls) == calls_before + 2
    # Unconfigured work after failure belongs outside the experiment directories.
    unconfigured_indexes = [
        path for path in tmp_path.rglob("index.json") if "experiments" not in path.parts
    ]
    assert len(unconfigured_indexes) == 2
    assert cached.mteb_model_meta.experiment_kwargs is None


def test_cache_identity_includes_both_sides(multimodal_inputs, tmp_path):
    corpus, _, metadata = multimodal_inputs
    queries = corpus.select([0]).remove_columns("id").add_column("id", ["q1"])
    metadata = metadata.model_copy(update={"category": "it2it"})
    encoder = RecordingEncoder()
    cached = CachedEmbeddingWrapper(encoder, tmp_path)
    configurations = [
        (["image"], ["text"]),
        (["text"], ["text"]),
        (["text"], ["image"]),
        (["image"], ["text"]),
    ]
    for query_selection, corpus_selection in configurations:
        wrapper = ModalityFilterWrapper(
            cached,
            query_modalities=query_selection,
            corpus_modalities=corpus_selection,
        )
        _run_search(wrapper, (corpus, queries, metadata))
    assert len(encoder.calls) == 6
    assert {call[1].category for call in encoder.calls} == {"i2t", "t2t", "t2i"}
    assert len(list(tmp_path.rglob("index.json"))) == 6


def test_loader_explicit_selection_overrides_saved_selection(multimodal_inputs):
    loader_calls = []

    def loader(model_name, revision=None, **kwargs: Any):
        loader_calls.append(kwargs)
        return RecordingSearch()

    saved_meta = _meta(
        loader=loader,
        experiment_kwargs={
            "query_modalities": ["text"],
            "corpus_modalities": ["image"],
            "custom_setting": 7,
        },
    )
    overridden = saved_meta.load_model(
        corpus_modalities=["text"], query_modalities=None
    )
    _run_search(overridden, multimodal_inputs)
    assert overridden.mteb_model_meta.experiment_kwargs == {
        "corpus_modalities": ["text"],
        "custom_setting": 7,
    }
    assert set(overridden.model.index_calls[0][0].column_names) & {"text", "image"} == {
        "text"
    }
    assert loader_calls == [{"custom_setting": 7}]
    assert saved_meta.experiment_kwargs["corpus_modalities"] == ["image"]
    assert saved_meta.experiment_kwargs["query_modalities"] == ["text"]


@pytest.mark.parametrize("selection", [[], ["unknown"], "image", ["audio"]])
def test_invalid_loader_selection_fails_before_loading_weights(selection):
    loader_calls = []

    def loader(model_name, revision=None, **kwargs: Any):
        loader_calls.append(kwargs)
        return RecordingSearch()

    meta = _meta(loader=loader)
    with pytest.raises(ValueError, match="modalit"):
        meta.load_model(corpus_modalities=selection)
    assert not loader_calls


def test_get_model_applies_modality_selection_after_loading(
    monkeypatch, multimodal_inputs
):
    import importlib

    loader_calls = []

    def loader(model_name, revision=None, **kwargs: Any):
        loader_calls.append(kwargs)
        return RecordingSearch()

    module = importlib.import_module("mteb.models.get_model_meta")
    monkeypatch.setattr(
        module, "get_model_meta", lambda *args, **kwargs: _meta(loader=loader)
    )
    model = mteb.get_model(
        "mteb/baseline-modality-selection", corpus_modalities=["image"]
    )
    _run_search(model, multimodal_inputs)
    assert isinstance(model, ModalityFilterWrapper)
    assert loader_calls == [{}]
    assert set(model.model.index_calls[0][0].column_names) == {"id", "image"}


def test_threaded_custom_search_preserves_embedding_cache_identity(
    multimodal_inputs, tmp_path
):
    from concurrent.futures import ThreadPoolExecutor

    encoder = RecordingEncoder()
    cached = CachedEmbeddingWrapper(encoder, tmp_path)

    class ThreadedSearch(RecordingSearch):
        def __init__(self):
            super().__init__()
            self.impl = SearchEncoderWrapper(cached)

        def index(self, corpus, **kwargs: Any):
            self.impl.index(corpus, **kwargs)

        def search(self, queries, **kwargs: Any):
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(self.impl.search, queries, **kwargs).result()

    base = ThreadedSearch()
    for selection in (["text"], ["image"], ["text"]):
        _run_search(
            ModalityFilterWrapper(base, corpus_modalities=selection),
            multimodal_inputs,
        )
    assert len(encoder.calls) == 4
    query_calls = [call for call in encoder.calls if call[0] == PromptType.query]
    assert {call[1].category for call in query_calls} == {"t2t", "t2i"}
    assert len(list(tmp_path.rglob("index.json"))) == 4


def test_nested_filter_inside_custom_search_is_rejected(multimodal_inputs):
    class NestedSearch(RecordingSearch):
        def __init__(self):
            super().__init__()
            self.inner = ModalityFilterWrapper(
                RecordingEncoder(), query_modalities=["text"]
            )

        def index(self, corpus, **kwargs: Any):
            self.inner.index(corpus, **kwargs)

    with pytest.raises(ValueError, match="Nested"):
        _run_search(
            ModalityFilterWrapper(NestedSearch(), corpus_modalities=["image"]),
            multimodal_inputs,
        )


def test_run_selection_does_not_serialize_as_task_metadata(multimodal_inputs):
    base = RecordingSearch()
    _run_search(
        ModalityFilterWrapper(base, corpus_modalities=["image"]), multimodal_inputs
    )
    metadata = base.index_calls[0][1]["task_metadata"]
    assert metadata._modality_filter_configuration == {"corpus_modalities": ["image"]}
    assert "_modality_filter_configuration" not in metadata.model_dump()
    assert multimodal_inputs[2]._modality_filter_configuration is None


def test_serialized_registry_experiment_restores_filtering():
    base = mteb.get_model("mteb/baseline-random-encoder")
    selected = ModalityFilterWrapper(base, corpus_modalities=["text"])
    saved = json.dumps(selected.mteb_model_meta.to_dict())
    restored_meta = ModelMeta.model_validate_json_resolved(saved)
    restored = restored_meta.load_model()
    assert isinstance(restored, ModalityFilterWrapper)
    assert (
        restored.mteb_model_meta.experiment_name
        == selected.mteb_model_meta.experiment_name
    )
    result = mteb.evaluate(restored, MockRetrievalTask(), cache=None, co2_tracker=False)
    assert result.task_results[0].main_score is not None


def test_cross_encoder_experiment_evaluates_reranking_task():
    base = RecordingCrossEncoder()
    wrapped = ModalityFilterWrapper(base, corpus_modalities=["text"])
    result = mteb.evaluate(wrapped, MockRerankingTask(), cache=None, co2_tracker=False)
    assert base.calls
    assert result.task_results[0].main_score is not None
    assert base.mteb_model_meta.experiment_kwargs is None
