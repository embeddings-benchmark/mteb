from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from mteb.benchmarks._create_table import SummaryTable
from mteb.benchmarks.benchmark import Benchmark
from mteb.leaderboard.table import apply_summary_styling_from_benchmark


@pytest.fixture
def summary_df():
    return pl.DataFrame(
        {
            "Rank (Borda)": [1, 2, 3, 4, 5],
            "Model": ["dominated", "small", "best", "unknown", "incomplete"],
            "Active Parameters (B)": [0.3, 0.1, 0.2, None, 0.0],
            "Mean (Task)": [0.60001, 0.6, 0.60002, 0.9, None],
            "Mean (TaskType)": [0.9, 0.8, 0.7, 0.6, None],
            "Release Date": ["2026-01-01"] * 5,
        }
    )


def _render(summary_df):
    benchmark = Mock(spec=Benchmark)
    benchmark.name = "synthetic"
    benchmark._create_summary_table.return_value = SummaryTable(df=summary_df)
    component, raw = apply_summary_styling_from_benchmark(benchmark, pl.DataFrame())
    displayed = pd.DataFrame(
        component.value["data"], columns=component.value["headers"]
    )
    return component, displayed, raw


def test_pareto_survives_summary_formatting(summary_df):
    component, displayed, raw = _render(summary_df)

    assert displayed.columns[:3].tolist() == ["Rank (Borda)", "Model", "Pareto"]
    assert displayed["Model"].tolist() == summary_df["Model"].to_list()
    assert displayed["Pareto"].tolist() == ["No", "Yes", "Yes", "N/A", "N/A"]
    # Scores that round to the same display value still use their full precision.
    assert displayed["Mean (Task)"].tolist()[:3] == [60.0, 60.0, 60.0]
    assert "Release Date" not in displayed
    assert component.pinned_columns == 2
    assert component.show_search == "filter"
    pd.testing.assert_frame_equal(raw, summary_df.to_pandas())


def test_pareto_recalculates_for_filtered_summary(summary_df):
    _, displayed, _ = _render(summary_df.filter(pl.col("Model") != "best"))
    assert displayed["Pareto"].tolist() == ["Yes", "Yes", "N/A", "N/A"]


@pytest.mark.parametrize(
    "missing_columns",
    [
        ["Mean (Task)"],
        ["Active Parameters (B)"],
        ["Mean (Task)", "Active Parameters (B)"],
    ],
)
def test_pareto_with_missing_columns(summary_df, missing_columns):
    _, displayed, raw = _render(summary_df.drop(missing_columns))
    assert displayed["Pareto"].tolist() == ["N/A"] * len(summary_df)
    assert "Pareto" not in raw


def test_pareto_with_all_null_values(summary_df):
    _, displayed, _ = _render(
        summary_df.with_columns(pl.lit(None).alias("Active Parameters (B)"))
    )
    assert displayed["Pareto"].tolist() == ["N/A"] * len(summary_df)
