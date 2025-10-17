import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_numeric_dtype

from mteb.benchmarks.benchmark import Benchmark
from mteb.results.benchmark_results import BenchmarkResults


def _borda_count(scores: pd.Series) -> pd.Series:
    n = len(scores)
    ranks = scores.rank(method="average", ascending=False)
    counts = n - ranks
    return counts


def _get_borda_rank(score_table: pd.DataFrame) -> pd.Series:
    borda_counts = score_table.apply(_borda_count, axis="index")
    mean_borda = borda_counts.sum(axis=1)
    return mean_borda.rank(method="min", ascending=False).astype(int)


def _format_scores(score: float) -> float:
    return round(score * 100, 2)


def _get_column_types(df: pd.DataFrame) -> list[str]:
    types = []
    for column_name in df.columns:
        if is_numeric_dtype(df[column_name]):
            types.append("number")
        else:
            types.append("str")
    return types


def _get_column_widths(df: pd.DataFrame) -> list[str]:
    # Please do not remove this function when refactoring.
    # Column width calculation seeminlgy changes regularly with Gradio releases,
    # and this piece of logic is good enough to quickly fix related issues.
    widths = []
    for column_name in df.columns:
        column_word_lengths = [len(word) for word in column_name.split()]
        if is_numeric_dtype(df[column_name]):
            value_lengths = [len(f"{value:.2f}") for value in df[column_name]]
        else:
            value_lengths = [len(str(value)) for value in df[column_name]]
        max_length = max(max(column_word_lengths), max(value_lengths))
        n_pixels = 25 + (max_length * 10)
        widths.append(f"{n_pixels}px")
    return widths


def _format_zero_shot(zero_shot_percentage: int):
    if zero_shot_percentage == -1:
        return "⚠️ NA"
    return f"{zero_shot_percentage:.0f}%"


def _create_light_green_cmap():
    cmap = plt.cm.get_cmap("Greens")
    num_colors = 256
    half_colors = np.linspace(0, 0.5, num_colors)
    half_cmap = [cmap(val) for val in half_colors]
    light_green_cmap = LinearSegmentedColormap.from_list(
        "LightGreens", half_cmap, N=256
    )
    return light_green_cmap


def apply_summary_styling_from_benchmark(
    benchmark_instance: Benchmark, benchmark_results: BenchmarkResults
) -> gr.DataFrame:
    """Apply styling to summary table created by the benchmark instance's _create_summary_table method.

    This supports polymorphism - different benchmark classes can have different table generation logic.

    Args:
        benchmark_instance: The benchmark instance
        benchmark_results: BenchmarkResults object containing model results (may be pre-filtered)

    Returns:
        Styled gr.DataFrame ready for display in the leaderboard
    """
    # Use the instance method to support polymorphism
    summary_df = benchmark_instance._create_summary_table(benchmark_results)

    # If it's a no-results DataFrame, return it as-is
    if "No results" in summary_df.columns:
        return gr.DataFrame(summary_df)

    # Apply the styling
    return _apply_summary_table_styling(summary_df)


def apply_per_task_styling_from_benchmark(
    benchmark_instance: Benchmark, benchmark_results: BenchmarkResults
) -> gr.DataFrame:
    """Apply styling to per-task table created by the benchmark instance's _create_per_task_table method.

    This supports polymorphism - different benchmark classes can have different table generation logic.

    Args:
        benchmark_instance: The benchmark instance
        benchmark_results: BenchmarkResults object containing model results (may be pre-filtered)

    Returns:
        Styled gr.DataFrame ready for display in the leaderboard
    """
    # Use the instance method to support polymorphism
    per_task_df = benchmark_instance._create_per_task_table(benchmark_results)

    # If it's a no-results DataFrame, return it as-is
    if "No results" in per_task_df.columns:
        return gr.DataFrame(per_task_df)

    # Apply the styling
    return _apply_per_task_table_styling(per_task_df)


def _apply_summary_table_styling(joint_table: pd.DataFrame) -> gr.DataFrame:
    """Apply styling to a raw summary DataFrame

    Returns:
        Styled gr.DataFrame ready for display in the leaderboard
    """
    excluded_columns = [
        "Rank (Borda)",
        "Rank",
        "Model",
        "Number of Parameters",
        "Embedding Dimensions",
        "Max Tokens",
        "Memory Usage (MB)",
    ]

    gradient_columns = [
        col for col in joint_table.columns if col not in excluded_columns
    ]
    light_green_cmap = _create_light_green_cmap()

    # Determine score columns (before formatting)
    score_columns = [
        col
        for col in joint_table.columns
        if col not in excluded_columns + ["Zero-shot"]
    ]

    numeric_data = joint_table.copy()

    # Format data for display
    if "Zero-shot" in joint_table.columns:
        joint_table["Zero-shot"] = joint_table["Zero-shot"].apply(_format_zero_shot)
    joint_table[score_columns] = joint_table[score_columns].map(_format_scores)

    joint_table_style = joint_table.style.format(
        {**dict.fromkeys(score_columns, "{:.2f}"), "Rank (Borda)": "{:.0f}"},
        na_rep="",
    )
    joint_table_style = joint_table_style.highlight_min(
        "Rank (Borda)", props="font-weight: bold"
    ).highlight_max(subset=score_columns, props="font-weight: bold")

    # Apply background gradients for each selected column
    for col in gradient_columns:
        if col in joint_table.columns:
            mask = numeric_data[col].notna()
            if col != "Zero-shot":
                gmap_values = numeric_data[col] * 100
                cmap = light_green_cmap
                joint_table_style = joint_table_style.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[mask, col],
                    gmap=gmap_values.loc[mask],
                )
            else:
                gmap_values = numeric_data[col]
                cmap = "RdYlGn"
                joint_table_style = joint_table_style.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[mask, col],
                    vmin=50,
                    vmax=100,
                    gmap=gmap_values.loc[mask],
                )

    column_types = _get_column_types(joint_table_style.data)
    # setting model name column to markdown
    if len(column_types) > 1:
        column_types[1] = "markdown"

    column_widths = _get_column_widths(joint_table_style.data)
    if len(column_widths) > 0:
        column_widths[0] = "100px"
    if len(column_widths) > 1:
        column_widths[1] = "250px"

    return gr.DataFrame(
        joint_table_style,
        datatype=column_types,
        interactive=False,
        pinned_columns=2,
        column_widths=column_widths,
        wrap=True,
        show_fullscreen_button=True,
        show_copy_button=True,
        show_search="filter",
    )


def _apply_per_task_table_styling(per_task: pd.DataFrame) -> gr.DataFrame:
    """Apply styling to a raw per-task DataFrame

    Returns:
        Styled gr.DataFrame ready for display in the leaderboard
    """
    task_score_columns = per_task.select_dtypes("number").columns
    per_task[task_score_columns] *= 100

    per_task_style = per_task.style.format(
        "{:.2f}", subset=task_score_columns, na_rep=""
    ).highlight_max(subset=task_score_columns, props="font-weight: bold")

    return gr.DataFrame(
        per_task_style,
        interactive=False,
        pinned_columns=1,
        show_fullscreen_button=True,
        show_copy_button=True,
        show_search="filter",
    )
