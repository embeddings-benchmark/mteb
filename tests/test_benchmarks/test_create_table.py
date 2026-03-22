"""Tests for leaderboard summary tables built from benchmark results."""

import pandas as pd

import mteb
from mteb import Benchmark
from mteb.benchmarks._create_table import _create_summary_table_from_benchmark_results
from mteb.results import BenchmarkResults, ModelResult, TaskResult


def test_mean_task_is_nan_when_benchmark_task_missing() -> None:
    """Incomplete coverage of benchmark tasks must not average only over present tasks."""
    sts12 = mteb.get_task("STS12")
    tr = TaskResult.from_task_results(
        sts12,
        scores={"test": {"default": {"main_score": 0.5}}},
        evaluation_time=1.0,
    )
    bench = Benchmark(
        name="sts12_sts13",
        tasks=mteb.get_tasks(["STS12", "STS13"]),
    )
    br = BenchmarkResults.model_construct(
        model_results=[
            ModelResult.model_construct(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_revision="rev",
                task_results=[tr],
            )
        ],
        benchmark=bench,
    )
    df = _create_summary_table_from_benchmark_results(br)
    assert pd.isna(df["Mean (Task)"].iloc[0])
