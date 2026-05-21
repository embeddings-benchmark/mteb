from __future__ import annotations

import numpy as np
import pytest

import mteb
from mteb import DBSFHybridSearch, RelativeScoreFusionHybridSearch, RRFHybridSearch
from tests.mock_tasks import MockRetrievalTask


def test_hybrid_search_init_and_meta():
    """Test the initialization, weight validation, and metadata generation of hybrid search wrappers."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)

    hybrid = DBSFHybridSearch(m1, m2)
    assert len(hybrid.models) == 2
    assert len(hybrid.wrapped_models) == 2
    assert len(hybrid.weights) == 2
    assert hybrid.weights == [0.5, 0.5]

    hybrid_weighted = DBSFHybridSearch(m1, m2, weights=[0.7, 0.3])
    assert hybrid_weighted.weights == [0.7, 0.3]

    meta = hybrid.mteb_model_meta
    assert (
        "hybrid-dbsfhybridsearch/baseline-random-encoder-baseline-random-encoder"
        in meta.name
    )

    with pytest.raises(
        ValueError, match="Length of weights must match the number of models"
    ):
        DBSFHybridSearch(m1, m2, weights=[0.5])

    with pytest.raises(
        TypeError,
        match="Expected a SearchProtocol, EncoderProtocol, or CrossEncoderProtocol",
    ):
        DBSFHybridSearch(m1, "not-a-model")


def test_dbsf_fusion_logic():
    """Verify that Distribution-Based Score Fusion (DBSF) normalizes and fuses scores correctly."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    hybrid = DBSFHybridSearch(m1, m2, weights=[0.6, 0.4])

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
    hybrid = RRFHybridSearch(m1, m2, weights=[0.5, 0.5], rrf_k=60)

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
    hybrid = RelativeScoreFusionHybridSearch(m1, m2, weights=[0.7, 0.3])

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

    for hybrid_cls in [
        DBSFHybridSearch,
        RRFHybridSearch,
        RelativeScoreFusionHybridSearch,
    ]:
        hybrid = hybrid_cls(m1, m2)
        task = MockRetrievalTask()
        results = mteb.evaluate(hybrid, task, cache=None)
        assert len(results) > 0
        assert results[0].scores is not None
