"""Unit tests for the polars-based leaderboard table builders.

Covers issue #5101: a model that only ran a subset of a task's splits (e.g.
`validation` + `test_iid` out of six) must not have those partial results
averaged into its task score as if it were fully evaluated — the score
should come back `None` instead, matching `build_task_scores`'s existing
`fully_covered` check in `mteb/api/aggregators.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from mteb.benchmarks._create_table import (
    _build_per_task_pivot,
    _create_summary_table,
    _incomplete_task_pairs,
)


def _long_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# Task "WebLINX" declares 4 splits; only "FullTask" declares just one
# (the common case, which must stay a no-op).
ROWS = [
    # Fully-evaluated model: all 4 splits of WebLINX, 1 subset each.
    {
        "model_name": "full",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "train",
        "score": 0.10,
    },
    {
        "model_name": "full",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "validation",
        "score": 0.20,
    },
    {
        "model_name": "full",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "test",
        "score": 0.90,
    },
    {
        "model_name": "full",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "test_iid",
        "score": 0.90,
    },
    {
        "model_name": "full",
        "task_name": "FullTask",
        "subset": "default",
        "split": "test",
        "score": 0.50,
    },
    # Partial model: only 2 of WebLINX's 4 splits, both happen to score high.
    {
        "model_name": "partial",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "validation",
        "score": 0.90,
    },
    {
        "model_name": "partial",
        "task_name": "WebLINX",
        "subset": "default",
        "split": "test_iid",
        "score": 0.90,
    },
    {
        "model_name": "partial",
        "task_name": "FullTask",
        "subset": "default",
        "split": "test",
        "score": 0.60,
    },
]


def test_incomplete_task_pairs_flags_partial_split_coverage():
    pairs = _incomplete_task_pairs(_long_df(ROWS)).sort("model_name", "task_name")
    assert pairs.to_dicts() == [{"model_name": "partial", "task_name": "WebLINX"}]


def test_incomplete_task_pairs_empty_when_every_model_fully_covered():
    rows = [r for r in ROWS if r["model_name"] == "full"]
    pairs = _incomplete_task_pairs(_long_df(rows))
    assert pairs.is_empty()


def test_build_per_task_pivot_nulls_partial_split_coverage():
    pivot = _build_per_task_pivot(_long_df(ROWS))
    assert pivot is not None
    per_task, task_cols = pivot
    assert set(task_cols) == {"WebLINX", "FullTask"}

    by_model = {r["model_name"]: r for r in per_task.to_dicts()}
    # Fully-evaluated model keeps its (mean-of-4-splits) score.
    assert by_model["full"]["WebLINX"] == pytest.approx((0.10 + 0.20 + 0.90 + 0.90) / 4)
    # Partial-coverage model's WebLINX cell is null, not the mean of its 2 splits —
    # it must not silently outrank "full" despite a higher raw split average.
    assert by_model["partial"]["WebLINX"] is None
    # Its single-split FullTask score is untouched (1 of 1 splits = fully covered).
    assert by_model["partial"]["FullTask"] == pytest.approx(0.60)


# `_attach_model_metadata` inner-joins against MODEL_REGISTRY, so the rows
# need real registered model names to survive into the summary frame.
FULL_MODEL = "mteb/baseline-random-encoder"
PARTIAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_create_summary_table_nulls_mean_task_on_partial_split_coverage(monkeypatch):
    fake_registry = {
        "WebLINX": SimpleNamespace(metadata=SimpleNamespace(type="Reranking")),
        "FullTask": SimpleNamespace(metadata=SimpleNamespace(type="Reranking")),
    }
    monkeypatch.setattr("mteb.benchmarks._create_table._TASKS_REGISTRY", fake_registry)
    rows_in = [
        {**r, "model_name": FULL_MODEL if r["model_name"] == "full" else PARTIAL_MODEL}
        for r in ROWS
    ]
    summary = _create_summary_table(_long_df(rows_in))
    assert not summary.is_empty
    rows = {r["Model"]: r for r in summary.df.to_dicts()}

    # `_create_summary_table` only surfaces task-*type*-level columns (both
    # tasks here are "Reranking"), plus the overall means — not raw per-task
    # columns (those live in the separate per-task table / `PerTaskTab`). Both
    # means are means-of-per-task-means: WebLINX averages to 0.525 across its
    # 4 splits, FullTask is 0.50, so the type/task mean is their average.
    full_mean = ((0.10 + 0.20 + 0.90 + 0.90) / 4 + 0.50) / 2
    assert rows[FULL_MODEL]["Reranking"] == pytest.approx(full_mean)
    assert rows[FULL_MODEL]["Mean (Task)"] == pytest.approx(full_mean)
    assert rows[FULL_MODEL]["Mean (TaskType)"] == pytest.approx(full_mean)

    # Partial coverage nulls WebLINX's cell in the underlying per-task pivot,
    # which (skipna=False semantics) cascades into the Reranking type mean and
    # Mean (Task) / Mean (TaskType) — a model can't outrank fully-evaluated
    # peers on the strength of an incomplete split set.
    assert rows[PARTIAL_MODEL]["Reranking"] is None
    assert rows[PARTIAL_MODEL]["Mean (Task)"] is None
    assert rows[PARTIAL_MODEL]["Mean (TaskType)"] is None
