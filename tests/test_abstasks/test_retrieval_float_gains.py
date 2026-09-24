"""Tests for retrieval tasks with continuous (float) relevance gains."""

from __future__ import annotations

import math
from typing import Any

import pytest
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval, _filter_queries_without_positives
from mteb.abstasks.retrieval_dataset_loaders import (
    RetrievalDatasetLoader,
    RetrievalSplitData,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks.retrieval import general_args
from mteb.models.model_meta import ModelMeta
from mteb.types import CorpusDatasetType

# the stub only needs to satisfy the runtime_checkable SearchProtocol's
# `mteb_model_meta` attribute
FIXED_SEARCH_META = ModelMeta.create_empty(
    overwrites={"name": "mock/fixed-score-search", "revision": "1"},
)


def _queries(*ids: str) -> Dataset:
    return Dataset.from_list([{"id": id_, "text": f"query {id_}"} for id_ in ids])


def test_filter_without_gains_drops_empty_qrels_only():
    relevant_docs = {"q1": {"d1": 1}, "q2": {}, "q3": {"d6": 0}}
    queries = _queries("q1", "q2", "q3", "q4")

    filtered_docs, filtered_queries = _filter_queries_without_positives(
        relevant_docs, queries
    )

    assert set(filtered_docs) == {"q1", "q3"}
    assert filtered_queries["id"] == ["q1", "q3"]


class FixedScoreSearch:
    """Duck-typed `SearchProtocol` returning fixed per-query score dicts."""

    def __init__(self, scores: dict[str, dict[str, float]]):
        self._scores = scores
        self.mteb_model_meta = FIXED_SEARCH_META

    def index(self, corpus: CorpusDatasetType, **kwargs: Any) -> None:
        return None

    def search(
        self,
        queries: Dataset,
        *,
        top_k: int,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        return {
            query_id: dict(list(self._scores[query_id].items())[:top_k])
            for query_id in queries["id"]
        }


class GainsRetrievalTask(AbsTaskRetrieval):
    """Tiny retrieval task with fixed data and float gains."""

    metadata = TaskMetadata(
        name="MockGainsRetrievalTask",
        main_score="ndcg_float_at_10",
        type="Retrieval",
        **general_args,
    )
    k_values = (1, 10)

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    corpus=Dataset.from_list(
                        [{"id": f"d{i}", "text": f"doc {i}"} for i in range(1, 7)]
                    ),
                    queries=_queries("q1", "q2", "q3"),
                    relevant_docs={"q1": {"d1": 1}, "q2": {"d6": 1}},
                    top_ranked=None,
                    gains={
                        # all-tied scores on q1 exercise the tie-class credit
                        "q1": {"d1": 1.0, "d2": 0.5, "d3": 0.0},
                        # all-zero gains: unscorable, scores 0.0 on the float metric
                        "q2": {"d6": 0.0},
                        # gain-only query: no human qrels, so it is filtered out
                        # before scoring and its gains are never used
                        "q3": {"d4": 0.8, "d5": 0.1},
                    },
                )
            }
        }
        self.data_loaded = True


def test_evaluate_with_gains() -> None:
    task = GainsRetrievalTask()
    model = FixedScoreSearch(
        {
            "q1": {"d1": 0.5, "d2": 0.5, "d3": 0.5},
            "q2": {"d6": 0.7},
            "q3": {"d4": 0.9, "d5": 0.2},
        }
    )

    scores = task.evaluate(model, split="test", encode_kwargs={})["default"]

    # q1: full tie block, group-mean gain (1 + 0.5 + 0) / 3 at every position
    q1 = (1.5 / 3) * (1 + 1 / math.log2(3) + 0.5)
    q1 /= 1 + 0.5 / math.log2(3)
    # q2: all-zero gains score 0.0 and stay in the denominator; q3 (gain-only,
    # no human qrels) is filtered out and must not enter the denominator
    expected_mean = round((q1 + 0.0) / 2, 5)
    assert scores["ndcg_float_at_10"] == pytest.approx(expected_mean)
    assert scores["main_score"] == scores["ndcg_float_at_10"]
    assert "nauc_ndcg_float_at_10_max" in scores
    # the integer-qrels metrics keep flowing alongside the float ones
    assert "ndcg_at_10" in scores


def test_evaluate_requires_gains_to_cover_qrel_queries() -> None:
    # queries with qrels but no gains entry would silently change the float
    # metric's denominator, so the metric raises rather than skipping them
    class GapsInGainsTask(GainsRetrievalTask):
        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            super().load_data(num_proc, **kwargs)
            del self.dataset["default"]["test"]["gains"]["q2"]

    task = GapsInGainsTask()
    model = FixedScoreSearch({"q1": {"d1": 0.9}, "q2": {"d6": 0.5}})

    with pytest.raises(KeyError, match="q2"):
        task.evaluate(model, split="test", encode_kwargs={})


def test_evaluate_without_gains_has_no_float_metrics() -> None:
    class PlainRetrievalTask(GainsRetrievalTask):
        metadata = TaskMetadata(
            name="MockPlainRetrievalTask",
            main_score="ndcg_at_10",
            type="Retrieval",
            **general_args,
        )

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            super().load_data(num_proc, **kwargs)
            self.dataset["default"]["test"].pop("gains")

    task = PlainRetrievalTask()
    model = FixedScoreSearch({"q1": {"d1": 0.9}, "q2": {"d6": 0.5}})

    scores = task.evaluate(model, split="test", encode_kwargs={})["default"]

    assert not any(key.startswith("ndcg_float") for key in scores)
    assert scores["main_score"] == scores["ndcg_at_10"]


def test_ignore_identical_ids_applies_to_the_float_metric_consistently() -> None:
    # ignore_identical_ids drops the query's own document from the ranking before
    # scoring. The float metric must drop it from the ideal ranking too; otherwise
    # the removed document's gain inflates the ideal DCG and a perfect ranking
    # cannot reach 1.0.
    class IdenticalIdsTask(GainsRetrievalTask):
        ignore_identical_ids = True

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            super().load_data(num_proc, **kwargs)
            split = self.dataset["default"]["test"]
            # q1 retrieves itself: "q1" is also a corpus id with a gain
            split["corpus"] = Dataset.from_list(
                [{"id": "q1", "text": "doc q1"}]
                + [{"id": f"d{i}", "text": f"doc {i}"} for i in range(1, 7)]
            )
            split["gains"] = {"q1": {"q1": 1.0, "d1": 0.5}, "q2": {"d6": 1.0}}

    model = FixedScoreSearch({"q1": {"q1": 0.9, "d1": 0.8}, "q2": {"d6": 0.7}})

    scores = IdenticalIdsTask().evaluate(model, split="test", encode_kwargs={})[
        "default"
    ]

    # without q1's own document, both queries are ranked in ideal gain order
    assert scores["ndcg_float_at_10"] == pytest.approx(1.0)


def test_load_gains_preserves_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    # the loader must NOT repeat the qrels int32 cast: gains are continuous
    # (e.g. sigmoid outputs), so 0.8 has to survive as 0.8
    import mteb.abstasks.retrieval_dataset_loaders as loaders

    monkeypatch.setattr(
        loaders, "get_dataset_config_names", lambda *args, **kwargs: ["gains"]
    )
    loader = RetrievalDatasetLoader(hf_repo="mock/repo", revision="1", split="test")
    monkeypatch.setattr(
        loader,
        "_load_dataset_split",
        lambda config, num_proc: Dataset.from_list(
            [
                {"query-id": "q1", "corpus-id": "d1", "gain": 0.8},
                {"query-id": "q1", "corpus-id": "d2", "gain": 0.0},
            ]
        ),
    )

    gains = loader._load_gains(num_proc=None, config="gains")

    assert gains == {"q1": {"d1": 0.8, "d2": 0.0}}
