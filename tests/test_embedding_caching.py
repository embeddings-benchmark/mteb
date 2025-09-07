from __future__ import annotations

import shutil

import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb import TaskMetadata
from mteb.models.model_implementations.cache_wrapper import CachedEmbeddingWrapper
from mteb.models.models_protocols import Encoder


class DummyModel(Encoder):
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def encode(self, inputs, **kwargs):
        self.call_count += 1
        return np.random.rand(len(inputs.dataset), self.embedding_dim).astype(  # noqa: NPY002
            np.float32
        )

    def random_other_function_returns_false(self):
        return False


class TestCachedEmbeddingWrapper:
    @pytest.fixture(scope="function")
    def cache_dir(self, tmp_path):
        cache_path = tmp_path / "test_cache"
        yield cache_path
        # Cleanup after test
        if cache_path.exists():
            shutil.rmtree(cache_path)

    def test_caching_functionality(self, cache_dir):
        # Create a dummy model
        dummy_model = DummyModel()
        dummy_task_metadata_query = TaskMetadata(
            name="DummyTaskQuery",
            description="Dummy task metadata",
            dataset={"path": "test", "revision": "test"},
            type="Classification",
            eval_langs=["eng-Latn"],
            main_score="accuracy",
        )
        dummy_task_metadata_corpus = dummy_task_metadata_query.copy()
        dummy_task_metadata_corpus.name = "DummyTaskCorpus"

        # Create the wrapper
        wrapped_model = CachedEmbeddingWrapper(dummy_model, cache_dir)

        # Simulate data
        queries = DataLoader(
            Dataset.from_dict(
                {
                    "text": [
                        "What is the effect of vitamin C on common cold?",
                        "How does exercise affect cardiovascular health?",
                    ]
                }
            )
        )
        corpus = DataLoader(
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
        query_embeddings1 = wrapped_model.encode(
            queries,
            task_metadata=dummy_task_metadata_query,
            hf_subset="test",
            hf_split="test",
        )
        corpus_embeddings1 = wrapped_model.encode(
            corpus,
            task_metadata=dummy_task_metadata_corpus,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 2  # One call for queries, one for corpus

        # Second call - should use cached embeddings
        query_embeddings2 = wrapped_model.encode(
            queries,
            task_metadata=dummy_task_metadata_query,
            hf_subset="test",
            hf_split="test",
        )
        corpus_embeddings2 = wrapped_model.encode(
            corpus,
            task_metadata=dummy_task_metadata_corpus,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 2  # No additional calls to the model

        # Verify that the embeddings are the same
        np.testing.assert_allclose(query_embeddings1, query_embeddings2)
        np.testing.assert_allclose(corpus_embeddings1, corpus_embeddings2)

        # Verify that cache files were created
        assert (cache_dir / "DummyTaskQuery" / "vectors.npy").exists()
        assert (cache_dir / "DummyTaskQuery" / "index.json").exists()
        assert (cache_dir / "DummyTaskCorpus" / "vectors.npy").exists()
        assert (cache_dir / "DummyTaskCorpus" / "index.json").exists()

        # Test with a new query - should use cache for existing queries and compute for new one
        new_queries = DataLoader(
            Dataset.from_dict({"text": ["What is the role of insulin in diabetes?"]})
        )
        query_embeddings3 = wrapped_model.encode(
            new_queries,
            task_metadata=dummy_task_metadata_query,
            hf_subset="test",
            hf_split="test",
        )

        assert dummy_model.call_count == 3  # One additional call for the new query
        assert query_embeddings3.shape == (1, dummy_model.embedding_dim)

        # try with a cached query only
        _ = wrapped_model.encode(
            queries,
            task_metadata=dummy_task_metadata_query,
            hf_subset="test",
            hf_split="test",
        )
        assert dummy_model.call_count == 3

        wrapped_model.close()  # delete to allow cleanup on Windows

    def test_other_functions_still_work(self, cache_dir):
        # Create a dummy model
        dummy_model = DummyModel()

        # Create the wrapper
        wrapped_model = CachedEmbeddingWrapper(dummy_model, cache_dir)

        # Call a function that is not wrapped
        result = wrapped_model.random_other_function_returns_false()

        assert result is False
        assert wrapped_model.call_count == 0

        wrapped_model.close()  # delete to allow cleanup on Windows
