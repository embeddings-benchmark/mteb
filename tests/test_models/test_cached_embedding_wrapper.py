import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import DataLoader

import mteb
from mteb import TaskMetadata
from mteb.abstasks import AbsTask
from mteb.models.cache_wrappers.cache_backend_protocol import CacheBackendProtocol
from mteb.models.cache_wrappers.cache_backends.faiss_cache import FaissCache
from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache
from mteb.models.cache_wrappers.cache_wrapper import CachedEmbeddingWrapper
from mteb.models.model_implementations.random_baseline import RandomEncoderBaseline
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType
from tests.mock_tasks import MockMultiChoiceTask, MockRetrievalTask


class DummyModel(RandomEncoderBaseline):
    call_count = 0

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
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


class TestCachedEmbeddingWrapper:
    @pytest.fixture(scope="function")
    def cache_dir(self, tmp_path):
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
    def test_caching_functionality(
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


@pytest.mark.parametrize(
    "task, model",
    [
        (
            MockMultiChoiceTask(),
            mteb.get_model("baseline/random-encoder-baseline"),
        ),  # ti2i
        (
            MockRetrievalTask(),
            mteb.get_model("baseline/random-encoder-baseline"),
        ),  # t2t
    ],
)
def test_wrapper_mock_tasks(task: AbsTask, model: EncoderProtocol, tmp_path: Path):
    cached_model = CachedEmbeddingWrapper(model, tmp_path)
    mteb.evaluate(cached_model, task, cache=None)
    assert len(list((tmp_path / task.metadata.name).glob("*"))) == 3
