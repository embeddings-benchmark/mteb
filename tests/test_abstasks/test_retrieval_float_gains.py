"""Retrieval tasks scored with float gains via `AbsTaskRetrieval.task_specific_scores`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from datasets import Dataset

from mteb._evaluators.retrieval_metrics import ndcg_float_scores
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.mocks.mock_tasks import MockRetrievalFloatGainsTask, MockRetrievalTask
from mteb.models.model_meta import ModelMeta
from mteb.tasks.retrieval.eng import bright_rcp_retrieval, nano_beir_rcp_retrieval
from mteb.tasks.retrieval.multilingual import vidore3_rcp_retrieval
from mteb.types import CorpusDatasetType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import RelevantDocumentsType

    GainsLoader = Callable[[TaskMetadata, str, str], dict[str, dict[str, float]]]

# each RCP task module carries its own copy of the gains loader; all of them are tested
GAINS_LOADERS = pytest.mark.parametrize(
    "load_float_gains",
    [
        nano_beir_rcp_retrieval.load_float_gains,
        bright_rcp_retrieval.load_float_gains,
        vidore3_rcp_retrieval.load_float_gains,
    ],
    ids=["nanobeir", "bright", "vidore"],
)

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


def _local_float_gains_task(
    path: Path, load_float_gains: GainsLoader
) -> AbsTaskRetrieval:
    """A task in exactly the shipped tasks' shape, reading a local dataset: the override
    loads the `gain` column of the subset's qrels and scores `ndcg_float_at_k` on it."""

    class LocalFloatGainsTask(AbsTaskRetrieval):
        metadata = MockRetrievalFloatGainsTask.metadata.model_copy(
            update={
                "name": "LocalFloatGainsTask",
                "dataset": {"path": str(path), "revision": "local"},
                "eval_splits": ["test"],
                "eval_langs": {SUBSET: ["eng-Latn"]},
            }
        )

        def task_specific_scores(
            self,
            scores: dict[str, dict[str, float]],
            qrels: RelevantDocumentsType,
            results: dict[str, dict[str, float]],
            hf_split: str,
            hf_subset: str,
        ) -> dict[str, float]:
            gains = load_float_gains(self.metadata, hf_subset, hf_split)
            return ndcg_float_scores(gains, results, self.k_values)

    return LocalFloatGainsTask()


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


def test_skip_first_result_is_not_applied_to_the_float_metric(
    task: MockRetrievalFloatGainsTask,
) -> None:
    """The documented contract (see `ndcg_float_scores`): the skip only affects the
    integer-qrels metrics; the float metric is computed on the unskipped ranking."""
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    without = task.evaluate(model, split="test", encode_kwargs={})["default"]
    task.skip_first_result = True
    with_skip = task.evaluate(model, split="test", encode_kwargs={})["default"]

    assert with_skip["ndcg_float_at_10"] == without["ndcg_float_at_10"]
    assert with_skip["ndcg_at_10"] != without["ndcg_at_10"]  # the skip applies there
    assert with_skip["main_score"] == with_skip["ndcg_float_at_10"]


def test_all_null_gains_query_scores_zero_and_stays_in_the_mean(
    task: MockRetrievalFloatGainsTask,
) -> None:
    """A query whose qrels gain entries are all null has no gains entry at all and scores 0.0."""

    class AllNullGainsTask(MockRetrievalFloatGainsTask):
        # q2's gain column is entirely null, so the loader yields no q2 entry
        float_gains = {
            **MockRetrievalFloatGainsTask.float_gains,
            "test": {"q1": {"d1": 0.9, "d2": 0.1}},
        }

    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = AllNullGainsTask().evaluate(model, split="test", encode_kwargs={})[
        "default"
    ]

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


SUBSET = "named"


def _write_local_dataset(
    root: Path, extra_qrels: Sequence[dict[str, Any]] = ()
) -> None:
    """A tiny parquet dataset in the RCP layout: configs ``{SUBSET}-corpus``, ``-queries``,
    ``-qrels`` (with a float ``gain`` column next to the integer ``score``) and ``-top_ranked``."""
    tables = {
        "corpus": [
            {"id": "d1", "title": "", "text": "first document"},
            {"id": "d2", "title": "", "text": "second document"},
        ],
        "queries": [{"id": "q1", "text": "a query"}, {"id": "q2", "text": "another"}],
        "qrels": [
            {"query-id": "q1", "corpus-id": "d1", "score": 1, "gain": 0.9},
            {"query-id": "q1", "corpus-id": "d2", "score": 0, "gain": 0.1},
            {"query-id": "q2", "corpus-id": "d1", "score": 0, "gain": 0.2},
            {"query-id": "q2", "corpus-id": "d2", "score": 1, "gain": 0.8},
            *extra_qrels,
        ],
        "top_ranked": [
            {"query-id": "q1", "corpus-ids": ["d1", "d2"]},
            {"query-id": "q2", "corpus-ids": ["d2", "d1"]},
        ],
    }
    configs = []
    for name, rows in tables.items():
        config = f"{SUBSET}-{name}"
        (root / config).mkdir(parents=True)
        Dataset.from_list(rows).to_parquet(str(root / config / "test.parquet"))
        configs.append(
            f"- config_name: {config}\n  data_files:\n"
            f"  - split: test\n    path: {config}/test.parquet\n"
        )
    (root / "README.md").write_text("---\nconfigs:\n" + "".join(configs) + "---\n")


@GAINS_LOADERS
def test_gains_are_read_from_the_subset_qrels(
    tmp_path: Path, load_float_gains: GainsLoader
) -> None:
    """The real loader path: the `gain` column of the subset's qrels config."""
    _write_local_dataset(tmp_path)
    task = _local_float_gains_task(tmp_path, load_float_gains)
    task.load_data()

    assert load_float_gains(task.metadata, SUBSET, "test") == {
        "q1": {"d1": 0.9, "d2": 0.1},
        "q2": {"d1": 0.2, "d2": 0.8},
    }
    model = FixedScoreSearch(
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.9, "d1": 0.1}}
    )
    scores = task.evaluate(model, split="test", encode_kwargs={})[SUBSET]
    assert scores["ndcg_float_at_10"] == pytest.approx(1.0)


@GAINS_LOADERS
def test_null_gains_are_skipped(tmp_path: Path, load_float_gains: GainsLoader) -> None:
    """A qrels row without a gain (``gain`` null) is not a candidate gain."""
    null_row = {"query-id": "q1", "corpus-id": "d9", "score": 0, "gain": None}
    _write_local_dataset(tmp_path, extra_qrels=[null_row])
    task = _local_float_gains_task(tmp_path, load_float_gains)

    assert load_float_gains(task.metadata, SUBSET, "test")["q1"] == {
        "d1": 0.9,
        "d2": 0.1,
    }
