"""Tests for the CustomGroup(ing) plumbing in mteb.api.aggregators/schemas.

Covers the scoped (subset-/split-narrowed) task-entry additions: schema
construction of `tasks`/`tasks_complete`, and `_build_summary_rows`'s
language-filter lenient-recompute merge (a scoped dimension must survive
untouched, not get wiped to `{}`).
"""

from __future__ import annotations

import polars as pl

import mteb
from mteb.api.aggregators import _build_summary_rows
from mteb.api.schemas import BenchmarkSchema
from mteb.benchmarks._create_table import SummaryTable
from mteb.benchmarks.benchmark import Benchmark, CustomGroup, CustomGrouping

FULL_MODEL = "mteb/baseline-random-encoder"


def _custom_grouping_schema(grouping: CustomGrouping):
    bench = Benchmark(
        name="mock_aggregators_schema",
        tasks=mteb.get_tasks(
            ["Banking77Classification", "CataloniaTweetClassification"]
        ),
        aggregations=(grouping,),
    )
    schema = BenchmarkSchema.from_benchmark(bench)
    (out,) = schema.custom_groupings
    return {g.label: g for g in out.groups}


def test_custom_group_schema_whole_task_only_group_is_complete():
    grouping = CustomGrouping(
        name="Dim",
        groups=(
            CustomGroup(
                label="Whole", tasks=mteb.get_tasks(["Banking77Classification"])
            ),
        ),
    )
    groups = _custom_grouping_schema(grouping)
    assert groups["Whole"].tasks == ["Banking77Classification"]
    assert groups["Whole"].tasks_complete is True


def test_custom_group_schema_scoped_only_group_is_incomplete_and_empty_tasks():
    scoped = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])
    grouping = CustomGrouping(
        name="Dim", groups=(CustomGroup(label="Scoped", tasks=[scoped]),)
    )
    groups = _custom_grouping_schema(grouping)
    assert groups["Scoped"].tasks == []
    assert groups["Scoped"].tasks_complete is False


def test_custom_group_schema_mixed_group_lists_whole_tasks_only_and_is_incomplete():
    scoped = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])
    whole = mteb.get_task("Banking77Classification")
    grouping = CustomGrouping(
        name="Dim", groups=(CustomGroup(label="Mixed", tasks=[scoped, whole]),)
    )
    groups = _custom_grouping_schema(grouping)
    assert groups["Mixed"].tasks == ["Banking77Classification"]
    assert groups["Mixed"].tasks_complete is False


def _summary_pl(cols: dict[str, list]) -> pl.DataFrame:
    return pl.DataFrame(cols)


def test_build_summary_rows_merges_lenient_recompute_not_overwrites():
    """A scoped dimension (absent from custom_group_task_to_label, per the
    has_scoped_refs gate) keeps its strict value under language_filtered;
    a whole-task dimension present in the mapping gets recomputed."""
    summary_pl = _summary_pl(
        {
            "Model": [FULL_MODEL],
            "Rank (Borda)": [1],
            "Mean (Task)": [0.5],
            "Mean (TaskType)": [0.5],
            "__cg__WholeDim::G1": [0.5],
            "__cg__ScopedDim::G2": [0.9],
        }
    )
    summary = SummaryTable(df=summary_pl)
    custom_group_cols_by_dim = {
        "WholeDim": ("__cg__WholeDim::G1",),
        "ScopedDim": ("__cg__ScopedDim::G2",),
    }
    # Only WholeDim is in this mapping -- mirrors the has_scoped_refs gate in
    # build_benchmark_summary excluding ScopedDim.
    custom_group_task_to_label = {"WholeDim": {"t1": "G1"}}
    per_task_rows = {FULL_MODEL: {"t1": 0.7}}

    rows = _build_summary_rows(
        summary_pl,
        summary,
        type_cols=[],
        per_task_rows=per_task_rows,
        trained_on_by_model={},
        task_to_type={},
        language_filtered=True,
        custom_group_cols_by_dim=custom_group_cols_by_dim,
        custom_group_task_to_label=custom_group_task_to_label,
    )

    assert len(rows) == 1
    row = rows[0]
    # WholeDim recomputed leniently from per_task_rows (t1=0.7 -> G1=0.7),
    # not left at its strict polars-column value (0.5).
    assert row.scores_by_custom_group["WholeDim"]["G1"] == 0.7
    # ScopedDim absent from custom_group_task_to_label -> untouched, keeps
    # its strict value (0.9) instead of being wiped to {} by an overwrite.
    assert row.scores_by_custom_group["ScopedDim"]["G2"] == 0.9


def test_build_summary_rows_strict_when_not_language_filtered():
    """language_filtered=False: no recompute at all, both dims keep their
    strict polars-column values."""
    summary_pl = _summary_pl(
        {
            "Model": [FULL_MODEL],
            "Rank (Borda)": [1],
            "Mean (Task)": [0.5],
            "Mean (TaskType)": [0.5],
            "__cg__WholeDim::G1": [0.5],
        }
    )
    summary = SummaryTable(df=summary_pl)
    custom_group_cols_by_dim = {"WholeDim": ("__cg__WholeDim::G1",)}

    rows = _build_summary_rows(
        summary_pl,
        summary,
        type_cols=[],
        per_task_rows={FULL_MODEL: {"t1": 0.7}},
        trained_on_by_model={},
        task_to_type={},
        language_filtered=False,
        custom_group_cols_by_dim=custom_group_cols_by_dim,
        custom_group_task_to_label={"WholeDim": {"t1": "G1"}},
    )

    assert rows[0].scores_by_custom_group["WholeDim"]["G1"] == 0.5
