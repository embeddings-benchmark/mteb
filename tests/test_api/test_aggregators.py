"""Tests for the CustomGroup(ing) plumbing in mteb.api.aggregators/schemas.

Covers the scoped (subset-/split-narrowed) task-entry additions: schema
construction of `tasks`/`tasks_complete`, and `_build_summary_rows` keeping
custom-group scores at their strict (all-or-nothing) value even under a
language filter, unlike `Mean(Task)`/`Mean(TaskType)`.
"""

from __future__ import annotations

import polars as pl
import pytest

import mteb
from mteb.benchmarks._create_table import SummaryTable
from mteb.benchmarks.benchmark import Benchmark, CustomGroup, CustomGrouping

FULL_MODEL = "mteb/baseline-random-encoder"


def _custom_grouping_schema(grouping: CustomGrouping):
    pytest.importorskip("fastapi")
    from mteb.api.schemas import BenchmarkSchema

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


def test_build_summary_rows_custom_groups_stay_strict_under_language_filter():
    """Custom-group scores are never recomputed leniently -- every task in a
    group must have a score, unlike Mean(Task)/Mean(TaskType), which average
    over whatever's present after a language filter."""
    pytest.importorskip("fastapi")
    from mteb.api.aggregators import _build_summary_rows

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
    # Even though the model has a post-filter score for t1 (which feeds
    # WholeDim::G1), that's irrelevant now -- custom groups always keep
    # their strict, polars-computed value.
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
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.scores_by_custom_group["WholeDim"]["G1"] == 0.5
    assert row.scores_by_custom_group["ScopedDim"]["G2"] == 0.9


def test_build_summary_rows_strict_when_not_language_filtered():
    pytest.importorskip("fastapi")
    from mteb.api.aggregators import _build_summary_rows

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
    )

    assert rows[0].scores_by_custom_group["WholeDim"]["G1"] == 0.5


def test_build_task_scores_ignores_splits_outside_eval_splits():
    pytest.importorskip("fastapi")
    from mteb.api.aggregators import build_task_scores

    task = "SprintDuplicateQuestions"
    assert list(mteb.get_task(task).metadata.eval_splits) == ["test"]
    other_model = "sentence-transformers/all-MiniLM-L6-v2"
    unified = pl.DataFrame(
        {
            "model_name": [FULL_MODEL, FULL_MODEL, other_model],
            "task_name": [task] * 3,
            "split": ["test", "validation", "test"],
            "subset": ["default"] * 3,
            "score": [0.6, 0.9, 0.7],
        }
    )

    result = build_task_scores(task, all_df=unified)

    assert result.splits == ["test"]
    scores = {row.model.name: row.score for row in result.rows}
    assert scores == {FULL_MODEL: 0.6, other_model: 0.7}
    assert all(
        "validation" not in s for row in result.rows for s in row.subset_scores.values()
    )


def test_build_task_scores_ignores_subsets_outside_hf_subsets():
    pytest.importorskip("fastapi")
    from mteb.api.aggregators import build_task_scores

    task = "HUMEWikipediaRerankingMultilingual"
    subsets = list(mteb.get_task(task).metadata.hf_subsets)
    assert "eng" not in subsets
    other_model = "sentence-transformers/all-MiniLM-L6-v2"
    unified = pl.DataFrame(
        {
            "model_name": [FULL_MODEL] * (len(subsets) + 1)
            + [other_model] * len(subsets),
            "task_name": [task] * (2 * len(subsets) + 1),
            "split": ["test"] * (2 * len(subsets) + 1),
            "subset": [*subsets, "eng", *subsets],
            "score": [0.6] * len(subsets) + [0.9] + [0.7] * len(subsets),
        }
    )
    result = build_task_scores(task, all_df=unified)

    assert result.subsets == sorted(subsets)
    scores = {row.model.name: row.score for row in result.rows}
    assert scores == pytest.approx({FULL_MODEL: 0.6, other_model: 0.7})


@pytest.mark.parametrize("stored_id", [None, "named-run"])
def test_variant_kwargs_follow_table_experiment_id(stored_id: str | None):
    pytest.importorskip("fastapi")
    from mteb.api.aggregators import _extract_variant_kwargs

    frame = pl.DataFrame(
        {"model_name": [FULL_MODEL], "experiments": [{"normalize": False}]}
    )
    if stored_id:
        frame = frame.with_columns(pl.lit(stored_id).alias("_experiment_id"))
    assert _extract_variant_kwargs(frame) == {
        (FULL_MODEL, stored_id or "normalize_False"): {"normalize": False}
    }


def test_task_detail_keeps_complementary_experiment_scores():
    pytest.importorskip("fastapi")
    from mteb.api import frames
    from mteb.api.aggregators import build_task_scores

    task = "CataloniaTweetClassification"
    raw = pl.DataFrame(
        {
            "model_name": [FULL_MODEL] * 4,
            "experiments": [None, None, {"normalize": False}, {"normalize": False}],
            "task_name": [task] * 4,
            "split": ["test"] * 4,
            "subset": ["catalan", "spanish"] * 2,
            "score": [0.9, 0.1, 0.1, 0.9],
        }
    )
    unified = frames._dedupe_unified(raw)
    rows = build_task_scores(task, all_df=unified).rows
    assert [
        (r.experiments, r.score, r.subset_scores["catalan"]["test"]) for r in rows
    ] == [(None, 0.5, 0.9), ({"normalize": False}, 0.5, 0.1)]


def test_coverage_requires_one_complete_run():
    pytest.importorskip("fastapi")
    from mteb.api import frames, routes
    from mteb.get_tasks import _TASKS_REGISTRY

    task = "CataloniaTweetClassification"
    other = "Banking77Classification"
    benchmark = Benchmark(name="coverage", tasks=mteb.get_tasks([task, other]))
    raw = pl.DataFrame(
        {
            "model_name": [FULL_MODEL] * 5,
            "_experiment_id": ["", "", "named", "named", "named"],
            "task_name": [task, task, task, task, other],
            "subset": ["catalan", "catalan", "spanish", "spanish", "default"],
            "split": ["test", "validation", "test", "validation", "test"],
            "score": [0.5] * 5,
        }
    )
    unified = frames._dedupe_unified(raw)
    # In the benchmark frame each run covers a different task.
    partial = raw.filter(
        pl.col("_experiment_id").eq("") | (pl.col("task_name") == other)
    )
    assert routes._count_full_models_by_benchmark(
        {"coverage": partial}, [benchmark]
    ) == {"coverage": 0}
    tasks = {task: _TASKS_REGISTRY[task], other: _TASKS_REGISTRY[other]}
    assert routes._count_full_models_by_task(unified, tasks) == {other: 1}


def test_experiment_kwargs_survive_summary_language_and_model_detail():
    pytest.importorskip("fastapi")
    from mteb.api import aggregators
    from mteb.api.schemas import BenchmarkSummarySchema

    task = "Banking77Classification"
    bench = Benchmark(name="named-run-test", tasks=mteb.get_tasks([task]))
    frame = pl.DataFrame(
        {
            "model_name": [FULL_MODEL] * 2,
            "experiments": [{"normalize": False}, None],
            "task_name": [task] * 2,
            "split": ["test"] * 2,
            "subset": ["default"] * 2,
            "language": [["eng-Latn"]] * 2,
            "score": [0.8, 0.2],
        }
    )
    summary = bench._create_summary_table(frame)
    rows = aggregators._build_summary_rows(
        summary.df,
        summary,
        [],
        {},
        {},
        {},
        False,
        variants_by_model=aggregators._extract_variant_kwargs(frame),
    )
    assert [(r.experiments, r.rank) for r in rows] == [
        ({"normalize": False}, 1),
        (None, 2),
    ]
    languages = aggregators._build_per_language_rows(frame)
    assert sorted(
        (r.scores_by_language["English"], r.experiments) for r in languages
    ) == [(0.2, None), (0.8, {"normalize": False})]
    payload = BenchmarkSummarySchema(
        benchmark_name=bench.name, task_types=[], tasks=[task], tasks_meta=[], rows=rows
    )
    detail = aggregators._model_scores_from_summaries(FULL_MODEL, [(bench, payload)])
    assert [(r.experiments, r.rank, r.mean_task) for r in detail.rows] == [
        ({"normalize": False}, 1, 0.8),
        (None, 2, 0.2),
    ]
