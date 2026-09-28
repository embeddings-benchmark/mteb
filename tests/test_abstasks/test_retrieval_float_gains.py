"""Retrieval tasks with float gains in the qrels (`AbsTaskRetrievalFloatGains`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from mteb.abstasks.retrieval_float_gains import AbsTaskRetrievalFloatGains
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


def test_ignore_identical_ids_drops_the_query_document_from_the_gains() -> None:
    # The evaluator drops each query's own document from the ranking in place. The
    # float metric must drop it from the gains too; otherwise its gain inflates the
    # ideal DCG and a perfect ranking of the remaining documents cannot reach 1.0.
    class IdenticalIdsTask(MockRetrievalFloatGainsTask):
        ignore_identical_ids = True
        float_gains = {
            **MockRetrievalFloatGainsTask.float_gains,
            "test": {"q1": {"q1": 1.0, "d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.8}},
        }

    model = FixedScoreSearch(
        {"q1": {"q1": 0.99, "d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = IdenticalIdsTask().evaluate(model, split="test", encode_kwargs={})[
        "default"
    ]

    assert scores["ndcg_float_at_10"] == pytest.approx(1.0)


def test_skip_first_result_is_rejected(task: MockRetrievalFloatGainsTask) -> None:
    task.skip_first_result = True
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    with pytest.raises(ValueError, match="skip_first_result"):
        task.evaluate(model, split="test", encode_kwargs={})


def test_gains_are_loaded_lazily_without_dataset_transform() -> None:
    # A caller that injects `task.dataset` directly skips `dataset_transform`; the
    # gains must then be loaded on first use instead of failing.
    task = MockRetrievalFloatGainsTask()
    task.load_data()
    del task._float_gains
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="test", encode_kwargs={})["default"]

    assert scores["ndcg_float_at_10"] == pytest.approx(1.0)


def _write_local_dataset(root: Path, subset: str | None) -> None:
    """A tiny parquet dataset whose qrels config carries a `gain` column.

    ``subset=None`` writes the default-subset layout (configs ``corpus``, ``queries``,
    ``qrels``, ``top_ranked``); otherwise every config is prefixed ``{subset}-``.
    """
    prefix = f"{subset}-" if subset else ""
    tables = {
        "corpus": [
            {"id": "d1", "title": "", "text": "first document"},
            {"id": "d2", "title": "", "text": "second document"},
        ],
        "queries": [{"id": "q1", "text": "a query"}, {"id": "q2", "text": "another"}],
        "qrels": [
            {
                "query-id": "q1",
                "corpus-id": "d1",
                "score": 1,
                "gain": 0.9,
                "theta": 1.0,
            },
            {
                "query-id": "q1",
                "corpus-id": "d2",
                "score": 0,
                "gain": 0.1,
                "theta": -1.0,
            },
            {
                "query-id": "q2",
                "corpus-id": "d1",
                "score": 0,
                "gain": 0.2,
                "theta": -0.5,
            },
            {
                "query-id": "q2",
                "corpus-id": "d2",
                "score": 1,
                "gain": 0.8,
                "theta": 0.5,
            },
        ],
        "top_ranked": [
            {"query-id": "q1", "corpus-ids": ["d1", "d2"]},
            {"query-id": "q2", "corpus-ids": ["d2", "d1"]},
        ],
    }
    if subset:
        # the named subset also excludes d2 for q1 (like BRIGHT's excluded_ids)
        tables["excluded"] = [{"query-id": "q1", "excluded-corpus-ids": ["d2"]}]
    configs = []
    for name, rows in tables.items():
        config = prefix + name
        (root / config).mkdir(parents=True)
        Dataset.from_list(rows).to_parquet(str(root / config / "test.parquet"))
        configs.append(
            f"- config_name: {config}\n  data_files:\n"
            f"  - split: test\n    path: {config}/test.parquet\n"
        )
    (root / "README.md").write_text("---\nconfigs:\n" + "".join(configs) + "---\n")


@pytest.mark.parametrize("subset", [None, "named"])
def test_gains_are_read_from_the_qrels_gain_column(
    tmp_path: Path, subset: str | None
) -> None:
    """The real loader path: the `gain` column of the (subset's) qrels config."""
    _write_local_dataset(tmp_path, subset)
    hf_subset = subset or "default"

    class LocalFloatGainsTask(AbsTaskRetrievalFloatGains):
        metadata = MockRetrievalFloatGainsTask.metadata.model_copy(
            update={
                "name": "LocalFloatGainsTask",
                "dataset": {"path": str(tmp_path), "revision": "local"},
                "eval_splits": ["test"],
                "eval_langs": {hf_subset: ["eng-Latn"]} if subset else ["eng-Latn"],
            }
        )

    task = LocalFloatGainsTask()
    task.load_data()

    assert task._float_gains[hf_subset]["test"] == {
        "q1": {"d1": 0.9, "d2": 0.1},
        "q2": {"d1": 0.2, "d2": 0.8},
    }
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="test", encode_kwargs={})[hf_subset]
    assert scores["ndcg_float_at_10"] == pytest.approx(1.0)


def test_gain_split_falls_back_to_the_only_split(tmp_path: Path) -> None:
    """Like the core loader: a requested split missing from a single-split config falls back."""
    _write_local_dataset(tmp_path, None)

    class LocalFloatGainsTask(AbsTaskRetrievalFloatGains):
        metadata = MockRetrievalFloatGainsTask.metadata.model_copy(
            update={
                "name": "LocalFloatGainsTask",
                "dataset": {"path": str(tmp_path), "revision": "local"},
                "eval_splits": ["test"],
                "eval_langs": ["eng-Latn"],
            }
        )

    gains = LocalFloatGainsTask()._load_float_gains("default", "dev", None)
    assert gains["q2"] == {"d1": 0.2, "d2": 0.8}


@pytest.mark.parametrize("subset", [None, "named"])
def test_full_corpus_retrieval_mode(tmp_path: Path, subset: str | None) -> None:
    """``rerank_top_ranked = False``: search the whole corpus minus excluded ids, no float metric."""
    _write_local_dataset(tmp_path, subset)
    hf_subset = subset or "default"

    class LocalRetrievalView(AbsTaskRetrievalFloatGains):
        rerank_top_ranked = False
        metadata = MockRetrievalFloatGainsTask.metadata.model_copy(
            update={
                "name": "LocalRetrievalView",
                "main_score": "ndcg_at_10",
                "dataset": {"path": str(tmp_path), "revision": "local"},
                "eval_splits": ["test"],
                "eval_langs": {hf_subset: ["eng-Latn"]} if subset else ["eng-Latn"],
            }
        )

    task = LocalRetrievalView()
    task.load_data()
    split = task.dataset[hf_subset]["test"]
    if subset:
        assert split["top_ranked"] == {"q1": ["d1"], "q2": ["d1", "d2"]}
    else:
        assert split["top_ranked"] is None
    assert sorted(split["corpus"]["id"]) == ["d1", "d2"]  # the corpus is not trimmed

    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="test", encode_kwargs={})[hf_subset]
    assert scores["main_score"] == scores["ndcg_at_10"]
    assert not any(key.startswith("ndcg_float") for key in scores)
