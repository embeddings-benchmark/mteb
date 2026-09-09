"""Unit tests for the polars-based leaderboard table builders.

Covers issue #5101: a model that only ran a subset of a task's splits must
not have those partial results averaged into its task score as if it were
fully evaluated — the score should come back `None` instead, matching
`build_task_scores`'s existing `fully_covered` check in
`mteb/api/aggregators.py`.
"""

from __future__ import annotations

import polars as pl
import pytest

import mteb
from mteb import ResultCache
from mteb.benchmarks._create_table import (
    _build_per_task_pivot,
    _create_summary_table,
    _incomplete_subset_pairs,
    _incomplete_task_pairs,
)
from mteb.benchmarks.benchmark import BenchmarkAggregation
from tests.conftest import _skip_if_datasets_too_old

FULL_MODEL = "mteb/baseline-random-encoder"
PARTIAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

POEM = "PoemSentimentClassification.v2"
BANKING77 = "Banking77Classification"
CATALONIA = "CataloniaTweetClassification"


def _task_frame(mock_mteb_cache: ResultCache, task_names: list[str]) -> pl.DataFrame:
    """Long results frame for `task_names`, straight from the mock cache."""
    tasks = mteb.get_tasks(task_names)
    mock_results = mock_mteb_cache.load_results()
    return mock_results.select_tasks(tasks)._to_results_df(tasks)


@_skip_if_datasets_too_old
def test_incomplete_task_pairs_flags_partial_split_coverage(
    mock_mteb_cache: ResultCache,
):
    pl_df = _task_frame(mock_mteb_cache, [POEM])
    pairs = _incomplete_task_pairs(pl_df).sort("model_name", "task_name")
    assert pairs.to_dicts() == [{"model_name": PARTIAL_MODEL, "task_name": POEM}]


@_skip_if_datasets_too_old
def test_build_per_task_pivot_nulls_partial_split_coverage(
    mock_mteb_cache: ResultCache,
):
    pl_df = _task_frame(mock_mteb_cache, [POEM, BANKING77])
    pivot = _build_per_task_pivot(pl_df)
    assert pivot is not None
    per_task, task_cols = pivot
    assert {POEM, BANKING77}.issubset(task_cols)

    by_model = {r["model_name"]: r for r in per_task.to_dicts()}
    # Fully-evaluated model keeps its (mean-of-2-splits) score.
    assert by_model[FULL_MODEL][POEM] == pytest.approx(0.251923)
    # Partial-coverage model's Poem cell is null — its real "validation"-only
    # score (0.440385) must not silently outrank the fully-evaluated peer.
    assert by_model[PARTIAL_MODEL][POEM] is None
    # Its single-split Banking77 score is untouched (1 of 1 splits = complete).
    assert by_model[PARTIAL_MODEL][BANKING77] == pytest.approx(0.800422)


@_skip_if_datasets_too_old
def test_create_summary_table_nulls_mean_task_on_partial_split_coverage(
    mock_mteb_cache: ResultCache,
):
    pl_df = _task_frame(mock_mteb_cache, [POEM, BANKING77])
    summary = _create_summary_table(pl_df)
    assert not summary.is_empty
    rows = {r["Model"]: r for r in summary.df.to_dicts()}

    # `_create_summary_table` only surfaces task-*type*-level columns (both
    # Poem and Banking77 are Classification), plus the overall means — not
    # raw per-task columns (those live in the separate per-task table /
    # `PerTaskTab`).
    full_mean = (0.251923 + 0.012532) / 2
    assert rows[FULL_MODEL]["Classification"] == pytest.approx(full_mean)
    assert rows[FULL_MODEL]["Mean (Task)"] == pytest.approx(full_mean)
    assert rows[FULL_MODEL]["Mean (TaskType)"] == pytest.approx(full_mean)

    # Partial coverage nulls Poem's cell in the underlying per-task pivot,
    # which (skipna=False semantics) cascades into the Classification type
    # column and Mean (Task) / Mean (TaskType) — a model can't outrank
    # fully-evaluated peers on the strength of an incomplete split set.
    assert rows[PARTIAL_MODEL]["Classification"] is None
    assert rows[PARTIAL_MODEL]["Mean (Task)"] is None
    assert rows[PARTIAL_MODEL]["Mean (TaskType)"] is None


@_skip_if_datasets_too_old
def test_incomplete_subset_pairs_flags_partial_split_coverage_per_subset(
    mock_mteb_cache: ResultCache,
):
    pl_df = _task_frame(mock_mteb_cache, [CATALONIA])
    pairs = _incomplete_subset_pairs(pl_df).sort("model_name", "task_name", "subset")
    assert pairs.to_dicts() == [
        {"model_name": PARTIAL_MODEL, "task_name": CATALONIA, "subset": "catalan"}
    ]


@_skip_if_datasets_too_old
def test_create_summary_table_nulls_mean_on_partial_split_coverage(
    mock_mteb_cache: ResultCache,
):
    pl_df = _task_frame(mock_mteb_cache, [CATALONIA, BANKING77])
    summary = _create_summary_table(
        pl_df,
        aggregations=(
            BenchmarkAggregation.MEAN_TASK,
            BenchmarkAggregation.MEAN_TASK_TYPE,
            BenchmarkAggregation.TASK_TYPES,
            BenchmarkAggregation.MEAN_SUBSET,
        ),
    )
    assert not summary.is_empty
    rows = {r["Model"]: r for r in summary.df.to_dicts()}

    # "full": spanish -> mean(0.333102, 0.337946), catalan -> mean(0.338905,
    # 0.339453), Banking77's one subset = 0.012532; Mean (Subset) is the
    # unweighted mean across all three (task, subset) pairs.
    full_spanish = (0.333102 + 0.337946) / 2
    full_catalan = (0.338905 + 0.339453) / 2
    assert rows[FULL_MODEL]["Mean (Subset)"] == pytest.approx(
        (full_spanish + full_catalan + 0.012532) / 3
    )

    catalonia_task_mean = (0.333102 + 0.337946 + 0.338905 + 0.339453) / 4
    full_task_mean = (catalonia_task_mean + 0.012532) / 2
    assert rows[FULL_MODEL]["Classification"] == pytest.approx(full_task_mean)
    assert rows[FULL_MODEL]["Mean (Task)"] == pytest.approx(full_task_mean)
    assert rows[FULL_MODEL]["Mean (TaskType)"] == pytest.approx(full_task_mean)

    assert rows[PARTIAL_MODEL]["Mean (Subset)"] is None
    assert rows[PARTIAL_MODEL]["Classification"] is None
    assert rows[PARTIAL_MODEL]["Mean (Task)"] is None
    assert rows[PARTIAL_MODEL]["Mean (TaskType)"] is None
