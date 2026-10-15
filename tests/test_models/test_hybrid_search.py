from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

import mteb
from mteb import HybridSearch
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.hybrid_wrappers import fuse_rrf


def test_hybrid_search_init_and_meta():
    """Test the initialization, weight validation, and metadata generation of hybrid search wrapper."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)

    hybrid = HybridSearch([m1, m2], fusion_strategy="dbsf")
    assert len(hybrid.wrapped_models) == 2
    assert len(hybrid.weights) == 2
    assert hybrid.weights == [0.5, 0.5]

    hybrid_weighted = HybridSearch([m1, m2], weights=[0.7, 0.3], fusion_strategy="dbsf")
    assert hybrid_weighted.weights == [0.7, 0.3]

    meta = hybrid.mteb_model_meta
    assert (
        meta.name
        == "mteb/baseline-hybrid-dbsf-baseline-random-encoder-baseline-random-encoder"
    )
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


@pytest.mark.parametrize("rrf_k", [0, 60])
@pytest.mark.parametrize("reverse_first", [False, True])
@pytest.mark.parametrize("reverse_second", [False, True])
def test_rrf_ties_are_independent_of_insertion_order(
    rrf_k, reverse_first, reverse_second
):
    scores1 = {"doc_a": 1.0, "doc_z": 1.0, "doc_top": 2.0}
    scores2 = {"doc_a": 0.0, "doc_z": 0.0, "doc_top": -1.0}
    if reverse_first:
        scores1 = dict(reversed(scores1.items()))
    if reverse_second:
        scores2 = dict(reversed(scores2.items()))

    fused = fuse_rrf([scores1, scores2], [0.7, 0.3], rrf_k=rrf_k)

    # Match retrieval evaluation's descending document ID tie break.
    assert fused == pytest.approx(
        {
            "doc_top": 0.7 / (rrf_k + 1) + 0.3 / (rrf_k + 3),
            "doc_z": 0.7 / (rrf_k + 2) + 0.3 / (rrf_k + 1),
            "doc_a": 0.7 / (rrf_k + 3) + 0.3 / (rrf_k + 2),
        }
    )


@pytest.mark.parametrize("fusion_strategy", ["rrf", "dbsf", "rsf"])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("reverse_models", [False, True])
def test_hybrid_search_breaks_fused_ties_before_top_k(
    monkeypatch, fusion_strategy, reverse_order, reverse_models
):
    models = [
        mteb.get_model("mteb/baseline-random-encoder", embed_dim=32) for _ in range(2)
    ]
    hybrid = HybridSearch(models, fusion_strategy=fusion_strategy)
    scores1 = {"doc_a": 2.0, "doc_z": 1.0}
    scores2 = {"doc_a": 1.0, "doc_z": 2.0}
    if reverse_order:
        scores1 = dict(reversed(scores1.items()))
        scores2 = dict(reversed(scores2.items()))
    if reverse_models:
        scores1, scores2 = scores2, scores1
    monkeypatch.setattr(
        hybrid.wrapped_models[0], "search", lambda **kwargs: {"q1": scores1}
    )
    monkeypatch.setattr(
        hybrid.wrapped_models[1], "search", lambda **kwargs: {"q1": scores2}
    )

    results = hybrid.search(
        queries=Dataset.from_list([{"id": "q1", "text": "query"}]),
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
        top_k=1,
        encode_kwargs={},
    )

    assert list(results["q1"]) == ["doc_z"]


def test_relative_score_fusion_logic():
    """Verify that Relative Score Fusion normalizes and fuses scores using min-max scaling."""
    m1 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    m2 = mteb.get_model("mteb/baseline-random-encoder", embed_dim=10)
    hybrid = HybridSearch([m1, m2], weights=[0.7, 0.3], fusion_strategy="rsf")

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
        "rsf",
    ]:
        hybrid = HybridSearch([m1, m2], fusion_strategy=strategy)
        task = MockRetrievalTask()
        results = mteb.evaluate(hybrid, task, cache=None)
        assert len(results) == 1
        assert results[0].get_score() == pytest.approx(0.63093)


def test_hybrid_search_with_cross_encoder():
    """Verify that hybrid search can fuse a CrossEncoder and a retriever model when top_ranked is provided."""
    retriever = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
    cross_encoder = mteb.get_model("mteb/baseline-random-cross-encoder")

    # Patch predict to act as a mock returning constant scores
    cross_encoder.predict = lambda inputs1, inputs2, **kwargs: np.ones(
        len(inputs2.dataset)
    )

    hybrid = HybridSearch([retriever, cross_encoder], fusion_strategy="dbsf")
    task = MockRetrievalTask()

    corpus = Dataset.from_list(
        [
            {"id": "doc1", "text": "document one"},
            {"id": "doc2", "text": "document two"},
        ]
    )
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
        top_ranked={"q1": ["doc1", "doc2"]},
    )
    assert list(res.keys()) == ["q1"]
    assert len(res["q1"]) == 2


def test_registered_hybrid_model_retrieval():
    """Verify that a registered hybrid model can be loaded and its metadata is correct."""
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    meta = mteb.get_model_meta(
        "mteb/baseline-hybrid-rrf-baseline-bm25s-multilingual-e5-small"
    )
    assert meta is not None
    assert meta.name == "mteb/baseline-hybrid-rrf-baseline-bm25s-multilingual-e5-small"
    assert meta.model_type == ["hybrid"]

    model = mteb.get_model(
        "mteb/baseline-hybrid-rrf-baseline-bm25s-multilingual-e5-small"
    )
    assert isinstance(model, HybridSearch)
    assert len(model.wrapped_models) == 2
