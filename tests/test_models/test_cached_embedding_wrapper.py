from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import mteb
from mteb._content_hashes import hash_item
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks import (
    MockMultiChoiceTask,
    MockRerankingTask,
    MockRetrievalTask,
    MockVideoAudioTextRetrievalVAT2T,
)
from mteb.models import MultiVectorSearchEncoderWrapper
from mteb.models.cache_wrappers.cache_backend_protocol import CacheBackendProtocol
from mteb.models.cache_wrappers.cache_backends.faiss_cache import FaissCache
from mteb.models.cache_wrappers.cache_backends.multi_vector_cache import (
    MultiVectorCache,
)
from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache
from mteb.models.cache_wrappers.cache_backends.sparse_cache import SparseCache
from mteb.models.cache_wrappers.cache_wrapper import (
    CachedEmbeddingWrapper,
    CachedMultiVectorWrapper,
)
from mteb.models.model_implementations.random_baseline import (
    RandomEncoderBaseline,
    RandomSparseEncoderBaseline,
    _key_to_multi_vector,
)
from mteb.models.models_protocols import EncoderProtocol
from mteb.similarity_functions import max_sim, pairwise_max_sim
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.types import Array, BatchedInput


class DummyModel(RandomEncoderBaseline):
    call_count = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mteb_model_meta = mteb.get_model_meta("mteb/baseline-random-encoder")

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        self.call_count += 1

        if task_metadata.name == "DummyFirstTask":
            # change inputs to simulate instruction processing
            old_inputs = inputs.dataset
            old_inputs = old_inputs.map(
                lambda x: {"text": x["text"] + " (first task processed)"}
            )
            inputs = DataLoader(old_inputs, batch_size=inputs.batch_size)
        embeddings = super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )
        if prompt_type is None:
            return embeddings
        prompt_offset = 1.0 if prompt_type is PromptType.query else 2.0
        return embeddings + prompt_offset


class TestCachedEmbeddingWrapper:
    @pytest.fixture
    def cache_dir(self, tmp_path: Path):
        cache_path = tmp_path / "test_cache"
        yield cache_path
        # Cleanup after test
        if cache_path.exists():
            shutil.rmtree(cache_path)

    @pytest.mark.parametrize(
        "cache_backend",
        [
            NumpyCache,
            FaissCache,
        ],
    )
    def test_caching_functionality(  # noqa: PLR0914
        self, cache_dir, cache_backend: type[CacheBackendProtocol]
    ):
        if cache_backend is FaissCache:
            try:
                import faiss  # noqa: F401
            except ImportError:
                pytest.skip("faiss is not installed")

        # Create a dummy model
        dummy_model = DummyModel("test_model", revision=None)
        first_task_metadata = TaskMetadata(
            name="DummyFirstTask",
            description="Dummy task metadata",
            dataset={"path": "test", "revision": "test"},
            type="Classification",
            eval_langs=["eng-Latn"],
            main_score="accuracy",
        )
        second_task_metadata = first_task_metadata.model_copy()
        second_task_metadata.name = "DummySecondTask"

        # Create the wrapper
        wrapped_model = CachedEmbeddingWrapper(
            dummy_model, cache_dir, cache_backend=cache_backend
        )

        # Simulate data
        first_task_texts = DataLoader(
            Dataset.from_dict(
                {
                    "text": [
                        "What is the effect of vitamin C on common cold?",
                        "How does exercise affect cardiovascular health?",
                    ]
                }
            )
        )
        second_task_texts = DataLoader(
            Dataset.from_dict(
                {
                    "text": [
                        "Vitamin C supplementation does not significantly reduce the incidence of common cold.",
                        "Regular exercise improves cardiovascular health by strengthening the heart and reducing blood pressure.",
                        "The impact of vitamin C on common cold duration is minimal according to recent studies.",
                    ]
                }
            )
        )

        # First call - should use the model to compute embeddings
        first_task_embeddings1 = wrapped_model.encode(
            first_task_texts,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        second_task_embeddings1 = wrapped_model.encode(
            second_task_texts,
            task_metadata=second_task_metadata,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 2  # One call for queries, one for corpus

        # Second call - should use cached embeddings
        first_task_embeddings2 = wrapped_model.encode(
            first_task_texts,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        second_task_embeddings2 = wrapped_model.encode(
            second_task_texts,
            task_metadata=second_task_metadata,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 2  # No additional calls to the model

        # Verify that the embeddings are the same
        np.testing.assert_allclose(first_task_embeddings1, first_task_embeddings2)
        np.testing.assert_allclose(second_task_embeddings1, second_task_embeddings2)

        if cache_backend is NumpyCache:
            # Verify that cache files were created
            assert (cache_dir / first_task_metadata.name / "vectors.npy").exists()
            assert (cache_dir / first_task_metadata.name / "index.json").exists()
            assert (cache_dir / second_task_metadata.name / "vectors.npy").exists()
            assert (cache_dir / second_task_metadata.name / "index.json").exists()
        else:
            assert (cache_dir / first_task_metadata.name / "index.json").exists()
            assert (cache_dir / first_task_metadata.name / "vectors.faiss").exists()
            assert (cache_dir / second_task_metadata.name / "index.json").exists()
            assert (cache_dir / second_task_metadata.name / "vectors.faiss").exists()

        # Pass first dataloader with second task metadata - should compute new embeddings
        first_task_embeddings_new_meta = wrapped_model.encode(
            first_task_texts,
            task_metadata=second_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        assert dummy_model.call_count == 3  # One additional call for new task metadata
        assert not np.allclose(first_task_embeddings1, first_task_embeddings_new_meta)

        # Test with a new query - should use cache for existing queries and compute for new one
        new_queries = DataLoader(
            Dataset.from_dict({"text": ["What is the role of insulin in diabetes?"]})
        )
        first_task_embeddings3 = wrapped_model.encode(
            new_queries,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 4  # One additional call for the new query
        assert first_task_embeddings3.shape == (1, dummy_model.embedding_dim)

        # try with a cached query only
        _ = wrapped_model.encode(
            first_task_texts,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        assert dummy_model.call_count == 4

        first_task_model_embeddings = wrapped_model.encode(
            first_task_texts,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )

        second_task_model_embeddings = wrapped_model.encode(
            second_task_texts,
            task_metadata=second_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        np.testing.assert_allclose(first_task_embeddings1, first_task_model_embeddings)
        np.testing.assert_allclose(
            second_task_embeddings1, second_task_model_embeddings
        )

        text_partly_cached = DataLoader(
            Dataset.from_dict(
                {
                    "text": [
                        "What is the effect of vitamin C on common cold?",
                        "New unseen query about health benefits of meditation.",
                        "What is the effect of vitamin C on common cold?",
                    ]
                }
            )
        )
        model_encode = dummy_model.encode(
            text_partly_cached,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        cached_model_encode = wrapped_model.encode(
            text_partly_cached,
            task_metadata=first_task_metadata,
            hf_subset="test",
            hf_split="test",
        )
        np.testing.assert_allclose(model_encode, cached_model_encode)

        wrapped_model.close()  # delete to allow cleanup on Windows

    def test_cache_isolated_by_prompt_type(self, cache_dir: Path):
        model = DummyModel("test_model", revision=None)
        wrapped_model = CachedEmbeddingWrapper(model, cache_dir)
        task_metadata = MockRetrievalTask().metadata
        inputs = DataLoader(
            Dataset.from_dict({"id": ["1"], "title": [""], "text": ["same input"]})
        )

        def cached_encode(prompt_type: PromptType):
            return wrapped_model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_subset="default",
                hf_split="test",
                prompt_type=prompt_type,
            )

        try:
            query_embeddings = cached_encode(PromptType.query)
            document_embeddings = cached_encode(PromptType.document)
            cached_query_embeddings = cached_encode(PromptType.query)
            cached_document_embeddings = cached_encode(PromptType.document)
        finally:
            wrapped_model.close()

        assert not np.array_equal(document_embeddings, query_embeddings)
        np.testing.assert_allclose(cached_query_embeddings, query_embeddings)
        np.testing.assert_allclose(cached_document_embeddings, document_embeddings)
        assert model.call_count == 2

    def test_cache_reused_by_new_wrapper(self, cache_dir: Path):
        model = DummyModel("test_model", revision=None)
        task_metadata = MockRetrievalTask().metadata
        inputs = DataLoader(Dataset.from_dict({"text": ["first", "second"]}))

        def encode_with_new_wrapper():
            wrapped_model = CachedEmbeddingWrapper(model, cache_dir)
            try:
                return wrapped_model.encode(
                    inputs,
                    task_metadata=task_metadata,
                    hf_subset="default",
                    hf_split="test",
                )
            finally:
                wrapped_model.close()

        embeddings = encode_with_new_wrapper()
        cached_embeddings = encode_with_new_wrapper()

        np.testing.assert_allclose(cached_embeddings, embeddings)
        assert model.call_count == 1

    def test_cache_isolated_by_experiment(self, cache_dir: Path):
        """Runs that produce different embeddings (e.g. a ``precision`` encode kwarg)
        must not share the embedding cache with the default run.

        Regression test for https://github.com/embeddings-benchmark/mteb/issues/5394
        """
        model = DummyModel("test_model", revision=None)
        wrapped_model = CachedEmbeddingWrapper(model, cache_dir)
        task_metadata = MockRetrievalTask().metadata
        inputs = DataLoader(Dataset.from_dict({"id": ["1"], "text": ["same input"]}))

        def cached_encode(**kwargs: Any):
            return wrapped_model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_subset="default",
                hf_split="test",
                **kwargs,
            )

        try:
            default_embeddings = cached_encode()
            int8_embeddings = cached_encode(precision="int8")
            cached_default = cached_encode()
            cached_int8 = cached_encode(precision="int8")
        finally:
            wrapped_model.close()

        # the int8 run re-encoded instead of reusing the default cache entry
        assert model.call_count == 2
        np.testing.assert_allclose(cached_default, default_embeddings)
        np.testing.assert_allclose(cached_int8, int8_embeddings)
        assert (cache_dir / task_metadata.name / "vectors.npy").exists()
        assert (
            cache_dir
            / task_metadata.name
            / "experiments"
            / "output_dtypes_int8"
            / "vectors.npy"
        ).exists()


@pytest.mark.parametrize("n_first", [1, 3])
@pytest.mark.parametrize("cache_backend", [NumpyCache, FaissCache])
def test_cache_reload_and_grow(
    tmp_path: Path, n_first: int, cache_backend: type[CacheBackendProtocol]
):
    if cache_backend is FaissCache:
        pytest.importorskip("faiss")

    items = [{"text": f"item {i}"} for i in range(n_first + 3)]
    vectors = np.random.default_rng(0).random((len(items), 4), dtype=np.float32)

    def open_cache() -> CacheBackendProtocol:
        cache = (
            NumpyCache(tmp_path, initial_vectors=1)
            if cache_backend is NumpyCache
            else cache_backend(tmp_path)
        )
        cache.load()
        return cache

    cache = open_cache()  # empty directory
    assert cache.get_vector(items[0]) is None
    cache.add(items[:n_first], vectors[:n_first])
    cache.save()
    cache.close()

    # Reopen and append without losing the previously cached vectors.
    cache = open_cache()
    cache.add(items[n_first:], vectors[n_first:])
    cache.save()
    cache.close()

    cache = open_cache()
    try:
        assert len(cache.hash_to_index) == len(items)
        for item, vector in zip(items, vectors, strict=True):
            np.testing.assert_array_equal(cache.get_vector(item), vector)
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("task", "model"),
    [
        (
            MockMultiChoiceTask(),
            mteb.get_model("mteb/baseline-random-encoder"),
        ),  # ti2i
        (
            MockRetrievalTask(),
            mteb.get_model("mteb/baseline-random-encoder"),
        ),  # t2t
    ],
)
def test_wrapper_mock_tasks(task: AbsTask, model: EncoderProtocol, tmp_path: Path):
    cached_model = CachedEmbeddingWrapper(model, tmp_path)
    try:
        results = mteb.evaluate(cached_model, task, cache=None)
    finally:
        cached_model.close()

    assert results[0].task_name == task.metadata.name


def test_wrapper_caches_multimodal_items(tmp_path: Path):
    """Items combining text, audio, and video together are cached and reused.

    Exercises `hash_item` combining all three modalities into one cache key
    (`MockVideoAudioTextRetrievalVAT2T` queries carry video+audio+text on the
    same row), not just single-modality items.
    """
    task = MockVideoAudioTextRetrievalVAT2T()
    try:
        task.load_data()
    except ImportError as error:
        pytest.skip(f"modality dependencies are not installed: {error}")

    dummy_model = DummyModel("multimodal_test_model", revision=None)
    wrapped_model = CachedEmbeddingWrapper(dummy_model, tmp_path)
    try:
        mteb.evaluate(wrapped_model, task, cache=None)
        first_run_call_count = dummy_model.call_count
        assert first_run_call_count > 0

        # Every item was cached on the first run, so re-evaluating the same
        # task must not trigger any further encode() calls.
        mteb.evaluate(wrapped_model, task, cache=None)
        assert dummy_model.call_count == first_run_call_count
    finally:
        wrapped_model.close()


class DummySparseModel(RandomSparseEncoderBaseline):
    """Returns sparse COO tensors, like `SparseEncoderWrapper` does for retrieval tasks."""

    call_count = 0

    def __init__(self) -> None:
        super().__init__("sparse_test_model", revision=None)
        self.mteb_model_meta = mteb.get_model_meta(
            "mteb/baseline-random-sparse-encoder"
        )

    def encode(self, inputs: DataLoader[BatchedInput], **kwargs: Any) -> Array:
        self.call_count += 1
        return torch.from_numpy(super().encode(inputs, **kwargs)).to_sparse()

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        assert embeddings1.is_sparse
        assert embeddings2.is_sparse
        return super().similarity(embeddings1.to_dense(), embeddings2.to_dense())


class DummyEncodeOnlyMultiVectorModel:
    """Multi-vector model with only `_encode`, like PyLate models (no `similarity` or `corpus_chunk_size`)."""

    call_count = 0

    def __init__(self) -> None:
        self.mteb_model_meta = mteb.get_model_meta("mteb/baseline-random-colbert")

    def _encode(self, inputs: DataLoader[BatchedInput], **kwargs: Any) -> Array:
        self.call_count += 1
        return [
            torch.from_numpy(
                _key_to_multi_vector(hash_item({"text": text}), 2 + len(text) % 5, 8)
            ).float()
            for text in inputs.dataset["text"]
        ]


class DummyMultiVectorModel(
    DummyEncodeOnlyMultiVectorModel, MultiVectorSearchEncoderWrapper
):
    """Search-only model returning a variable number of token embeddings per input."""

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        return max_sim(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        return pairwise_max_sim(embeddings1, embeddings2)


def _cached_dense(path: Path) -> CachedEmbeddingWrapper:
    return CachedEmbeddingWrapper(DummyModel("dense_test_model", revision=None), path)


def _cached_sparse(path: Path) -> CachedEmbeddingWrapper:
    return CachedEmbeddingWrapper(DummySparseModel(), path, cache_backend=SparseCache)


def _cached_multi_vector(path: Path) -> CachedMultiVectorWrapper:
    return CachedMultiVectorWrapper(DummyMultiVectorModel(), path)


def _cached_encode_only_multi_vector(path: Path) -> CachedMultiVectorWrapper:
    return CachedMultiVectorWrapper(DummyEncodeOnlyMultiVectorModel(), path)  # type: ignore[arg-type]


def _sparse_vectors(n: int) -> torch.Tensor:
    dense = torch.from_numpy(np.random.default_rng(0).random((n, 10), dtype=np.float32))
    dense[dense < 0.7] = 0
    dense[0] = 0  # an all-zero row
    return dense.to_sparse()


def _multi_vectors(n: int) -> list[torch.Tensor]:
    # bfloat16, to check that cached embeddings keep the dtype they were added with
    return [torch.rand(i + 1, 4).to(torch.bfloat16) for i in range(n)]


@pytest.mark.parametrize(
    ("cache_backend", "vectors"),
    [
        (SparseCache, _sparse_vectors(6)),
        (SparseCache, _sparse_vectors(6).to_dense().numpy()),
        (MultiVectorCache, _multi_vectors(6)),
    ],
    ids=["sparse", "sparse-dense-input", "multi-vector"],
)
def test_ragged_cache_reload_and_grow(
    tmp_path: Path,
    cache_backend: type[SparseCache | MultiVectorCache],
    vectors: Array | list[torch.Tensor],
):
    items = [{"text": f"item {i}"} for i in range(len(vectors))]

    def open_cache() -> SparseCache | MultiVectorCache:
        cache = cache_backend(tmp_path, initial_rows=1)
        cache.load()
        return cache

    cache = open_cache()
    assert cache.get_vector(items[0]) is None
    cache.add(items, vectors)
    cache.save()
    cache.close()

    # Reopen and add again: cached items are skipped, nothing is lost
    cache = open_cache()
    cache.add(items, vectors)
    cache.save()
    cache.close()

    cache = open_cache()
    assert len(cache.hash_to_range) == len(items)
    for i, item in enumerate(items):
        cached, vector = cache.get_vector(item), vectors[i]
        # Cached vectors come back in the form they were added: sparse, dense or of the same dtype
        assert type(cached) is type(vector)
        torch.testing.assert_close(
            torch.as_tensor(cached).to_dense(), torch.as_tensor(vector).to_dense()
        )
    cache.close()


@pytest.mark.parametrize(
    ("cache_backend", "vectors", "other_vectors", "error"),
    [
        (
            SparseCache,
            _sparse_vectors(1),
            _sparse_vectors(1).to_dense().numpy(),
            "stores sparse vectors",
        ),
        (SparseCache, np.ones((1, 4)), np.ones((1, 6)), "dimension 4"),
        (MultiVectorCache, [np.ones((2, 4))], [np.ones((2, 6))], r"\(num_tokens, 4\)"),
    ],
    ids=["sparse-kind", "sparse-dimension", "multi-vector-dimension"],
)
def test_ragged_cache_rejects_incompatible_vectors(
    tmp_path: Path,
    cache_backend: type[SparseCache | MultiVectorCache],
    vectors: Array | list[np.ndarray],
    other_vectors: Array | list[np.ndarray],
    error: str,
):
    cache = cache_backend(tmp_path)
    cache.add([{"text": "a"}], vectors)
    with pytest.raises(ValueError, match=error):
        cache.add([{"text": "b"}], other_vectors)
    assert {"text": "b"} not in cache
    cache.close()


def test_ragged_cache_writes_metadata_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A save interrupted while writing the metadata leaves no index entries pointing at unsaved rows."""
    cache = MultiVectorCache(tmp_path)
    cache.add([{"text": "a"}], [np.ones((2, 4))])
    monkeypatch.setattr(json, "dump", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        cache.save()
    assert not cache.index_file.exists()
    cache.close()


@pytest.mark.parametrize(
    "make_cached_model",
    [_cached_sparse, _cached_multi_vector],
    ids=["sparse", "multi-vector"],
)
def test_wrapper_combines_cached_and_new_vectors(
    tmp_path: Path,
    make_cached_model: Callable[
        [Path], CachedEmbeddingWrapper | CachedMultiVectorWrapper
    ],
):
    """A partly cached batch returns the cached and newly encoded vectors in input order."""
    texts = [f"text number {i}" for i in range(6)]

    def encode(model: object, batch: list[str]) -> Array:
        encode_fn = model.encode if hasattr(model, "encode") else model._encode
        return encode_fn(
            DataLoader(Dataset.from_dict({"text": batch})),
            task_metadata=MockRetrievalTask().metadata,
            hf_split="test",
            hf_subset="default",
        )

    cached_model = make_cached_model(tmp_path)
    try:
        expected = encode(cached_model._model, texts)
        encode(cached_model, texts[1::2])
        partly_cached = encode(cached_model, texts)
        fully_cached = encode(cached_model, texts)
    finally:
        cached_model.close()

    for result in (partly_cached, fully_cached):
        assert type(result) is type(expected)
        if isinstance(result, torch.Tensor) and result.is_sparse:
            assert result.is_coalesced()
        for i in range(len(texts)):
            vector, expected_vector = result[i], expected[i]
            # Cached vectors have the same type as newly encoded ones, so they can be mixed
            assert type(vector) is type(expected_vector)
            torch.testing.assert_close(
                torch.as_tensor(vector).to_dense(),
                torch.as_tensor(expected_vector).to_dense(),
            )


@pytest.mark.parametrize(
    ("make_model", "make_cached_model", "task"),
    [
        (
            lambda: DummyModel("dense_test_model", revision=None),
            _cached_dense,
            MockRetrievalTask(),
        ),
        (DummySparseModel, _cached_sparse, MockRetrievalTask()),
        (DummyMultiVectorModel, _cached_multi_vector, MockRetrievalTask()),
        (DummyMultiVectorModel, _cached_multi_vector, MockRerankingTask()),
        # Models with only `_encode`, like PyLate models, are searched with MaxSim
        (DummyMultiVectorModel, _cached_encode_only_multi_vector, MockRetrievalTask()),
        (DummyMultiVectorModel, _cached_encode_only_multi_vector, MockRerankingTask()),
    ],
    ids=[
        "dense",
        "sparse",
        "multi-vector-retrieval",
        "multi-vector-reranking",
        "encode-only-multi-vector-retrieval",
        "encode-only-multi-vector-reranking",
    ],
)
def test_wrapper_matches_uncached_scores(
    tmp_path: Path,
    make_model: Callable[[], Any],
    make_cached_model: Callable[
        [Path], CachedEmbeddingWrapper | CachedMultiVectorWrapper
    ],
    task: AbsTask,
):
    """Cached runs score like uncached ones (e.g. corpus titles aren't joined twice) and reuse the cache."""
    uncached = mteb.evaluate(make_model(), task, cache=None)

    cached_model = make_cached_model(tmp_path)
    try:
        first = mteb.evaluate(cached_model, task, cache=None)
        first_run_call_count = cached_model._model.call_count
        second = mteb.evaluate(cached_model, task, cache=None)
        assert cached_model._model.call_count == first_run_call_count
        assert uncached[0].scores == first[0].scores == second[0].scores
    finally:
        cached_model.close()


def test_cached_embedding_wrapper_rejects_multi_vector_model(tmp_path: Path):
    with pytest.raises(ValueError, match="CachedMultiVectorWrapper"):
        CachedEmbeddingWrapper(DummyEncodeOnlyMultiVectorModel(), tmp_path)  # type: ignore[arg-type]
