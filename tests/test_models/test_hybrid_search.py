from __future__ import annotations

import numpy as np
import pytest

import mteb
from mteb import HybridSearch
from tests.mock_tasks import MockRetrievalTask


def test_hybrid_search_init_and_meta():
    """Test the initialization, weight validation, and metadata generation of hybrid search wrapper."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)

    hybrid = HybridSearch([m1, m2], fusion_strategy="dbsf")
    assert len(hybrid.models) == 2
    assert len(hybrid.wrapped_models) == 2
    assert len(hybrid.weights) == 2
    assert hybrid.weights == [0.5, 0.5]

    hybrid_weighted = HybridSearch([m1, m2], weights=[0.7, 0.3], fusion_strategy="dbsf")
    assert hybrid_weighted.weights == [0.7, 0.3]

    meta = hybrid.mteb_model_meta
    assert "hybrid-dbsf/baseline-random-encoder-baseline-random-encoder" in meta.name
    assert meta.model_type == ["hybrid"]

    with pytest.raises(
        ValueError, match="Length of weights must match the number of models"
    ):
        HybridSearch([m1, m2], weights=[0.5], fusion_strategy="dbsf")

    with pytest.raises(
        TypeError,
        match="Expected a SearchProtocol, EncoderProtocol, or CrossEncoderProtocol",
    ):
        HybridSearch([m1, "not-a-model"], fusion_strategy="dbsf")

    with pytest.raises(
        ValueError, match="At least two models must be provided for hybrid search"
    ):
        HybridSearch([m1], fusion_strategy="dbsf")

    with pytest.raises(ValueError, match="sub_model_top_k must be greater than 0"):
        HybridSearch([m1, m2], sub_model_top_k=0, fusion_strategy="dbsf")


def test_dbsf_fusion_logic():
    """Verify that Distribution-Based Score Fusion (DBSF) normalizes and fuses scores correctly."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    hybrid = HybridSearch([m1, m2], weights=[0.6, 0.4], fusion_strategy="dbsf")

    scores1 = {"doc1": 1.0, "doc2": 2.0, "doc3": 3.0}
    scores2 = {"doc1": 10.0, "doc2": 10.0}

    fused = hybrid.fuse([scores1, scores2])

    mu = np.mean([1.0, 2.0, 3.0])
    sigma = np.std([1.0, 2.0, 3.0])
    denom = 6 * sigma

    norm1_doc1 = (1.0 - (mu - 3 * sigma)) / denom
    expected_doc1 = 0.6 * norm1_doc1 + 0.2

    norm1_doc3 = (3.0 - (mu - 3 * sigma)) / denom
    expected_doc3 = 0.6 * norm1_doc3

    assert pytest.approx(fused["doc1"]) == expected_doc1
    assert pytest.approx(fused["doc2"]) == 0.5
    assert pytest.approx(fused["doc3"]) == expected_doc3


def test_rrf_fusion_logic():
    """Verify that Reciprocal Rank Fusion (RRF) computes rank reciprocal scores correctly."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    hybrid = HybridSearch([m1, m2], weights=[0.5, 0.5], fusion_strategy="rrf", rrf_k=60)

    scores1 = {"doc1": 10.0, "doc2": 5.0}
    scores2 = {"doc2": 8.0, "doc1": 2.0}

    fused = hybrid.fuse([scores1, scores2])
    expected_doc1 = 0.5 * (1 / 61) + 0.5 * (1 / 62)
    expected_doc2 = 0.5 * (1 / 62) + 0.5 * (1 / 61)

    assert pytest.approx(fused["doc1"]) == expected_doc1
    assert pytest.approx(fused["doc2"]) == expected_doc2


def test_relative_score_fusion_logic():
    """Verify that Relative Score Fusion normalizes and fuses scores using min-max scaling."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    hybrid = HybridSearch(
        [m1, m2], weights=[0.7, 0.3], fusion_strategy="relative-score-fusion"
    )

    scores1 = {
        "doc1": 1.0,
        "doc2": 2.0,
        "doc3": 5.0,
    }
    scores2 = {"doc1": 10.0, "doc2": 10.0}

    fused = hybrid.fuse([scores1, scores2])

    assert pytest.approx(fused["doc1"]) == 0.15
    assert pytest.approx(fused["doc2"]) == 0.325
    assert pytest.approx(fused["doc3"]) == 0.7


def test_hybrid_search_e2e_retrieval():
    """Verify that all hybrid search wrappers can successfully evaluate a retrieval task."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)

    for strategy in [
        "dbsf",
        "rrf",
        "relative-score-fusion",
    ]:
        hybrid = HybridSearch([m1, m2], fusion_strategy=strategy)
        task = MockRetrievalTask()
        results = mteb.evaluate(hybrid, task, cache=None)
        assert len(results) > 0
        assert results[0].scores is not None


def test_hybrid_search_with_cross_encoder():
    """Verify that hybrid search can fuse a CrossEncoder and a retriever model even when top_ranked is None."""
    retriever = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    cross_encoder = mteb.get_model("mteb/baseline-random-cross-encoder")

    hybrid = HybridSearch([retriever, cross_encoder], fusion_strategy="dbsf")
    task = MockRetrievalTask()
    results = mteb.evaluate(hybrid, task, cache=None)
    assert len(results) > 0
    assert results[0].scores is not None

    with pytest.raises(
        ValueError,
        match="CrossEncoder sub-models require top_ranked documents for reranking",
    ):
        hybrid_only_ce = HybridSearch(
            [cross_encoder, cross_encoder], fusion_strategy="dbsf"
        )
        hybrid_only_ce.search(
            queries=[{"id": "q1", "text": "query"}],
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="default",
            top_k=5,
            encode_kwargs={},
            top_ranked=None,
        )


def test_candidate_trimming():
    """Verify that candidates generated from multiple retrievers are trimmed to sub_top_k."""
    from mteb.models import SearchCrossEncoderWrapper

    class MockRetriever:
        mteb_model_meta = None

        def index(self, *args, **kwargs):
            pass

        def search(self, queries, top_k, **kwargs):  # noqa: PLR6301
            return {"q1": {f"doc_{i}": float(i) for i in range(10)}}

    class MockSearchCrossEncoderWrapper(SearchCrossEncoderWrapper):
        mteb_model_meta = None

        def __init__(self):
            pass

        def search(self, queries, top_k, top_ranked, **kwargs):  # noqa: PLR6301
            assert top_ranked is not None
            assert len(top_ranked["q1"]) == top_k
            return {"q1": dict.fromkeys(top_ranked["q1"], 1.0)}

    retriever1 = MockRetriever()
    retriever2 = MockRetriever()
    cross_encoder = MockSearchCrossEncoderWrapper()

    hybrid = HybridSearch(
        [retriever1, retriever2, cross_encoder],
        sub_model_top_k=3,
        fusion_strategy="dbsf",
    )
    from tests.mock_tasks import MockRetrievalTask

    task = MockRetrievalTask()
    res = hybrid.search(
        queries=[{"id": "q1", "text": "query"}],
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={},
        top_ranked=None,
    )
    assert res is not None
