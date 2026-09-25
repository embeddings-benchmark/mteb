"""Retrieval tasks with float gains in the qrels (`AbsTaskRetrievalFloatGains`)."""

from __future__ import annotations

import math
from typing import Any

import pytest
from datasets import Dataset

from mteb.mocks.mock_tasks import MockRetrievalFloatGainsTask, MockRetrievalTask
from mteb.models.model_meta import ModelMeta
from mteb.types import CorpusDatasetType

FIXED_SEARCH_META = ModelMeta.create_empty(
    overwrites={"name": "mock/fixed-score-search", "revision": "1"},
)


class FixedScoreSearch:
    """Duck-typed `SearchProtocol` returning fixed per-query score dicts."""

    def __init__(self, scores: dict[str, dict[str, float]]):
        self._scores = scores
        self.mteb_model_meta = FIXED_SEARCH_META

    def index(self, corpus: CorpusDatasetType, **kwargs: Any) -> None:
        return None

    def search(
        self, queries: Dataset, *, top_k: int, **kwargs: Any
    ) -> dict[str, dict[str, float]]:
        return {query_id: dict(self._scores[query_id]) for query_id in queries["id"]}


def _ndcg(credited: list[float], gains: list[float]) -> float:
    dcg = sum(g / math.log2(r + 2) for r, g in enumerate(credited))
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(sorted(gains, reverse=True)))
    return dcg / idcg if idcg else 0.0


@pytest.fixture
def task() -> MockRetrievalFloatGainsTask:
    return MockRetrievalFloatGainsTask()


def test_float_gains_are_the_main_score(task: MockRetrievalFloatGainsTask) -> None:
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d1": 0.9, "d2": 0.1}}
    )
    scores = task.evaluate(model, split="test", encode_kwargs={})["default"]

    q1 = _ndcg([0.9, 0.1], [0.9, 0.1])  # ideal order
    q2 = _ndcg([0.2, 0.8], [0.2, 0.8])  # inverted order
    assert scores["ndcg_float_at_10"] == pytest.approx(round((q1 + q2) / 2, 5))
    assert scores["main_score"] == scores["ndcg_float_at_10"]
    assert "nauc_ndcg_float_at_10_max" in scores
    # the integer-qrels metrics are computed alongside, unchanged
    assert "ndcg_at_10" in scores


def test_ties_credit_the_group_mean_gain(task: MockRetrievalFloatGainsTask) -> None:
    model = FixedScoreSearch(
        {"q1": {"d1": 0.5, "d2": 0.5}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="ties", encode_kwargs={})["default"]

    q1 = _ndcg([0.5, 0.5], [0.9, 0.1])  # both tied docs credited (0.9 + 0.1) / 2
    q2 = _ndcg([0.8, 0.2], [0.2, 0.8])
    assert scores["ndcg_float_at_10"] == pytest.approx(round((q1 + q2) / 2, 5))


def test_all_zero_gains_score_zero_and_stay_in_the_mean(
    task: MockRetrievalFloatGainsTask,
) -> None:
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="zero_gains", encode_kwargs={})["default"]

    assert scores["ndcg_float_at_10"] == pytest.approx(round((1.0 + 0.0) / 2, 5))


def test_integer_metrics_match_a_plain_retrieval_task() -> None:
    """Adding float gains must not change any integer-qrels metric."""
    model_scores = {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d1": 0.9, "d2": 0.1}}
    with_gains = MockRetrievalFloatGainsTask().evaluate(
        FixedScoreSearch(model_scores), split="test", encode_kwargs={}
    )["default"]
    plain = MockRetrievalTask().evaluate(
        FixedScoreSearch(model_scores), split="test", encode_kwargs={}
    )["default"]

    for key, value in plain.items():
        if key not in {"main_score", "hf_subset", "languages"}:
            # nAUC is NaN on a two-document mock; NaN == NaN counts as a match
            assert with_gains[key] == value or (
                math.isnan(with_gains[key]) and math.isnan(value)
            ), key
    assert not any(key.startswith("ndcg_float") for key in plain)


def test_corpus_is_restricted_to_top_ranked(task: MockRetrievalFloatGainsTask) -> None:
    task.load_data()
    split = task.dataset["default"]["test"]
    split["top_ranked"] = {"q1": ["d1"], "q2": ["d1"]}
    task.dataset_transform()

    assert task.dataset["default"]["test"]["corpus"]["id"] == ["d1"]
