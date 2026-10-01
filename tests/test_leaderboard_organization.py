"""Organization grouping must select complete rows from the filtered summary."""

from dataclasses import replace
from html.parser import HTMLParser

import polars as pl
import pytest

from mteb.benchmarks._create_table import SummaryTable
from mteb.leaderboard.organization import (
    _cell_text,
    group_summary_by_organization,
    organization_summary_html,
)


@pytest.fixture
def summary() -> SummaryTable:
    return SummaryTable(
        df=pl.DataFrame(
            {
                "Rank (Borda)": [1, 2, 3, 4],
                "Model": ["alpha/small", "alpha/large", "beta/base", "beta/missing"],
                "Mean (Task)": [0.6, 0.8, 0.7, None],
                "Retrieval": [0.9, 0.5, 0.7, None],
            }
        )
    )


def test_representative_is_whole_row_not_borda_or_column_max(summary):
    original = summary.df.clone()
    groups = group_summary_by_organization(summary)
    assert [org for org, _ in groups] == ["alpha", "beta"]
    representative = groups[0][1].iloc[0]
    assert representative["Model"] == "alpha/large"
    assert representative["Retrieval"] == 0.5
    assert representative["Rank (Borda)"] == 2
    assert groups[1][1]["Model"].tolist() == ["beta/base", "beta/missing"]
    assert summary.df.equals(original)


def test_filtering_changes_representative_and_removes_organizations(summary):
    filtered = replace(summary, df=summary.df.filter(pl.col("Model") == "alpha/small"))
    groups = group_summary_by_organization(filtered)
    assert len(groups) == 1
    assert groups[0][1].iloc[0]["Model"] == "alpha/small"


def test_benchmark_declared_metric_is_used(summary):
    groups = group_summary_by_organization(
        replace(summary, primary_metric_col="Retrieval")
    )
    assert groups[0][1].iloc[0]["Model"] == "alpha/small"


def test_ties_and_missing_organizations_are_deterministic():
    summary = SummaryTable(
        df=pl.DataFrame(
            {
                "Model": ["org/z", "standalone", "org/a", "other"],
                "Mean (Task)": [0.5, None, 0.5, float("nan")],
            }
        )
    )
    groups = group_summary_by_organization(summary)
    assert [org for org, _ in groups] == ["org", "other", "standalone"]
    assert groups[0][1]["Model"].tolist() == ["org/a", "org/z"]


@pytest.mark.parametrize("sentinel", [True, False])
def test_empty_summary(sentinel):
    summary = SummaryTable(df=pl.DataFrame(), is_empty=sentinel)
    assert group_summary_by_organization(summary) == []
    assert "No results" in organization_summary_html(summary)


def test_expansion_includes_all_models_and_escapes_metadata(summary):
    summary = replace(
        summary,
        df=summary.df.with_columns(
            pl.col("Model").str.replace("alpha", '<script>alert("x")</script>')
        ),
    )
    html = organization_summary_html(summary)
    assert html.count("<details>") == 2
    assert html.count("<summary>") == 2
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "beta/base" in html and "beta/missing" in html
    assert "Mean (Task)" in html
    HTMLParser().feed(html)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("Mean (Task)", 0.81234, "81.23"),
        ("Mean (Task)", None, "—"),
        ("Mean (Task)", float("nan"), "—"),
        ("Rank (Borda)", 3, "3"),
        ("Zero-shot", -1, "⚠️ NA"),
        ("Zero-shot", 80, "80%"),
        ("Total Parameters (B)", 0.123, "0.123"),
        ("Embedding Dimensions", 768, "768"),
    ],
)
def test_cell_formatting(column, value, expected):
    assert _cell_text(column, value) == expected


def test_summary_views_preserve_experiment_text(summary):
    from unittest.mock import Mock

    from mteb.leaderboard.table import apply_summary_styling_from_benchmark

    summary = replace(
        summary, df=summary.df.with_columns(pl.lit("dim=128").alias("_experiment_id"))
    )
    benchmark = Mock(name="benchmark")
    benchmark._create_summary_table.return_value = summary
    table, raw, grouped = apply_summary_styling_from_benchmark(
        benchmark, pl.DataFrame()
    )
    assert "dim=128" in grouped
    assert raw["_experiment_id"].tolist() == ["dim=128"] * 4
    assert "_experiment_id" in table.value["headers"]
