from __future__ import annotations

import shutil

import numpy as np
import pytest

from mteb.evaluation import CachedEmbeddingWrapper


class DummyModel:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def encode_queries(self, queries):
        self.call_count += 1
        return np.random.rand(len(queries), self.embedding_dim)

    def encode_corpus(self, corpus):
        self.call_count += 1
        return np.random.rand(len(corpus), self.embedding_dim)

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

        # Create the wrapper
        wrapped_model = CachedEmbeddingWrapper(dummy_model, cache_dir)

        # Simulate data
        queries = [
            "What is the effect of vitamin C on common cold?",
            "How does exercise affect cardiovascular health?",
        ]
        corpus = [
            "Vitamin C supplementation does not significantly reduce the incidence of common cold.",
            "Regular exercise improves cardiovascular health by strengthening the heart and reducing blood pressure.",
            "The impact of vitamin C on common cold duration is minimal according to recent studies.",
        ]

        # First call - should use the model to compute embeddings
        query_embeddings1 = wrapped_model.encode_queries(queries)
        corpus_embeddings1 = wrapped_model.encode_corpus(corpus)

        assert dummy_model.call_count == 2  # One call for queries, one for corpus

        # Second call - should use cached embeddings
        query_embeddings2 = wrapped_model.encode_queries(queries)
        corpus_embeddings2 = wrapped_model.encode_corpus(corpus)

        assert dummy_model.call_count == 2  # No additional calls to the model

        # Verify that the embeddings are the same
        np.testing.assert_array_equal(query_embeddings1, query_embeddings2)
        np.testing.assert_array_equal(corpus_embeddings1, corpus_embeddings2)

        # Verify that cache files were created
        assert (cache_dir / "query_cache" / "vectors.npy").exists()
        assert (cache_dir / "query_cache" / "index.json").exists()
        assert (cache_dir / "corpus_cache" / "vectors.npy").exists()
        assert (cache_dir / "corpus_cache" / "index.json").exists()

        # Test with a new query - should use cache for existing queries and compute for new one
        new_queries = queries + ["What is the role of insulin in diabetes?"]
        query_embeddings3 = wrapped_model.encode_queries(new_queries)

        assert dummy_model.call_count == 3  # One additional call for the new query
        assert query_embeddings3.shape == (3, dummy_model.embedding_dim)
        np.testing.assert_array_equal(query_embeddings3[:2], query_embeddings2)

    def test_other_functions_still_work(self, cache_dir):
        # Create a dummy model
        dummy_model = DummyModel()

        # Create the wrapper
        wrapped_model = CachedEmbeddingWrapper(dummy_model, cache_dir)

        # Call a function that is not wrapped
        result = wrapped_model.random_other_function_returns_false()

        assert result is False
        assert wrapped_model.call_count == 0
