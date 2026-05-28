from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset

import mteb
from mteb import HybridSearch, HybridSearchProtocol
from tests.mock_tasks import MockRetrievalTask


def test_hybrid_search_init_and_meta():
    """Test the initialization, weight validation, and metadata generation of hybrid search wrapper."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)

    hybrid = HybridSearch([m1, m2], fusion_strategy="dbsf")
    assert isinstance(hybrid, HybridSearchProtocol)
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
        assert len(results) == 1
        assert results[0].get_score() == pytest.approx(0.63093)


def test_hybrid_search_with_cross_encoder():
    """Verify that hybrid search can fuse a CrossEncoder and a retriever model even when top_ranked is None."""
    retriever = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    cross_encoder = mteb.get_model("mteb/baseline-random-cross-encoder")

    hybrid = HybridSearch([retriever, cross_encoder], fusion_strategy="dbsf")
    task = MockRetrievalTask()
    results = mteb.evaluate(hybrid, task, cache=None)
    assert len(results) == 1
    assert results[0].get_score() == pytest.approx(0.81546)

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
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    cross_encoder = mteb.get_model("mteb/baseline-random-cross-encoder")

    # Patch predict to act as a mock returning constant scores
    cross_encoder.predict = lambda inputs1, inputs2, **kwargs: np.ones(
        len(inputs2.dataset)
    )

    hybrid = HybridSearch(
        [m1, m2, cross_encoder],
        sub_model_top_k=3,
        fusion_strategy="dbsf",
    )

    corpus = Dataset.from_list(
        [
            {"id": "doc1", "text": "document one"},
            {"id": "doc2", "text": "document two"},
            {"id": "doc3", "text": "document three"},
            {"id": "doc4", "text": "document four"},
            {"id": "doc5", "text": "document five"},
        ]
    )

    task = MockRetrievalTask()
    hybrid.index(
        corpus=corpus,
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        encode_kwargs={},
    )

    res = hybrid.search(
        queries=Dataset.from_list([{"id": "q1", "text": "query"}]),
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        top_k=2,
        encode_kwargs={},
        top_ranked=None,
    )
    assert list(res.keys()) == ["q1"]
    assert len(res["q1"]) == 2
    assert res["q1"]["doc4"] == pytest.approx(0.5748068685695726)
    assert res["q1"]["doc3"] == pytest.approx(0.4209643094972072)


def test_registered_hybrid_model_retrieval():
    """Verify that the registered hybrid-bm25s-e5-small model can be loaded and its metadata is correct."""
    meta = mteb.get_model_meta("mteb/hybrid-bm25s-e5-small")
    assert meta is not None
    assert meta.name == "mteb/hybrid-bm25s-e5-small"
    assert meta.model_type == ["hybrid"]

    with patch("mteb.models.model_meta.ModelMeta._check_requirements"):
        model = mteb.get_model("mteb/hybrid-bm25s-e5-small")
        assert isinstance(model, HybridSearch)
        assert len(model.models) == 2
        assert len(model.wrapped_models) == 2
