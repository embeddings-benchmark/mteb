from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import mteb
from mteb.load_results import load_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_benchmarks():
    """Get all available benchmark names."""
    return [b.name for b in mteb.get_benchmarks()]


def save_dataframe(
    df: pd.DataFrame,
    output_path: str,
):
    """Save a DataFrame to the specified format based on file extension.

    Args:
        df: The DataFrame to save
        output_path: Path for the output file, extension determines format

    Returns:
        str: The full path to the saved file
    """
    ext = Path(output_path).suffix.lower()
    fallback_path = str(Path(output_path).with_suffix(".csv"))

    def warn_and_fallback(reason: str):
        """Logs a warning and saves the DataFrame as CSV instead."""
        logger.warning(f"{reason}. Defaulting to CSV format: {fallback_path}")
        df.to_csv(fallback_path, index=False)
        return fallback_path

    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".xlsx":
        try:
            df.to_excel(output_path, index=False)
        except ImportError:
            return warn_and_fallback(
                "openpyxl not installed. Please install with 'pip install mteb[xlsx]' to save as Excel."
            )
    elif ext == ".md":
        try:
            with open(output_path, "w") as f:
                f.write(df.to_markdown(index=False))
        except ImportError:
            return warn_and_fallback(
                "tabulate not installed. Please install with 'pip install mteb[markdown]' to save as Markdown."
            )
    else:
        return warn_and_fallback(
            f"Unsupported file extension: {ext}, defaulting to CSV"
        )

    return output_path


def create_comparison_table(
    results_folder: str,
    model_names: list[str] | None = None,
    benchmark_name: str | None = None,
    output_path: str | None = None,
    aggregation_level: Literal["subset", "split", "task"] = "task",
) -> pd.DataFrame:
    """Create comparison tables for MTEB models.

    Args:
        results_folder: Path to the results folder
        model_names: List of model names to include (default: None, which means all available models)
        benchmark_name: Name of the benchmark (optional)
        output_path: Path to save the output tables
        aggregation_level: Level of aggregation for results ('subset', 'split', or 'task')
                          - 'subset': Results for each subset within each split for each task
                          - 'split': Results aggregated over subsets for each split for each task
                          - 'task': Results aggregated over subsets and splits for each task

    Returns:
        result_df: DataFrame with aggregated results
    """
    if model_names:
        logger.info(f"Creating comparison table for models: {', '.join(model_names)}")
    else:
        logger.info("Creating comparison table for all available models")

    logger.info(f"Using aggregation level: {aggregation_level}")

    # Load results
    benchmark_results = load_results(
        results_repo=results_folder,
        only_main_score=True,
        require_model_meta=False,
        models=model_names,
    )

    # Filter by benchmark if specified
    if benchmark_name:
        logger.info(f"Filtering tasks for benchmark: {benchmark_name}")
        benchmark = next(
            (b for b in mteb.get_benchmarks() if b.name == benchmark_name), None
        )
        if not benchmark:
            raise ValueError(
                f"Benchmark '{benchmark_name}' not found. Available: {get_available_benchmarks()}"
            )

        benchmark_results_filtered = benchmark.load_results(
            base_results=benchmark_results
        ).join_revisions()
    else:
        logger.info("Using all available tasks for the specified models")
        benchmark_results_filtered = benchmark_results.join_revisions()

    # Check if we have any results
    if not benchmark_results_filtered.model_results or not any(
        model_result.task_results
        for model_result in benchmark_results_filtered.model_results
    ):
        logger.warning("No results found for the specified models and benchmark")
        return pd.DataFrame()

    # Get detailed scores
    scores_data = []
    for model_result in benchmark_results_filtered.model_results:
        model_name = model_result.model_name
        for task_result in model_result.task_results:
            task_name = task_result.task_name
            for split, scores_list in task_result.scores.items():
                for score_item in scores_list:
                    scores_data.append(
                        {
                            "model_name": model_name,
                            "task_name": task_name,
                            "split": split,
                            "subset": score_item.get("hf_subset", "default"),
                            "score": score_item.get("main_score", 0.0) * 100,
                        }
                    )

    if not scores_data:
        logger.warning("No scores found for the specified models and benchmark")
        return pd.DataFrame()

    scores_df = pd.DataFrame(scores_data)

    # Create the appropriate table based on aggregation level
    if aggregation_level == "subset":
        # For subset level, show raw data at task/split/subset level (no aggregation)
        pivot_df = scores_df.pivot_table(
            index=["task_name", "split", "subset"],
            columns="model_name",
            values="score",
            aggfunc="mean",
        ).reset_index()

    elif aggregation_level == "split":
        # For split level, aggregate across subsets for each task/split combination
        agg_df = (
            scores_df.groupby(["model_name", "task_name", "split"])["score"]
            .mean()
            .reset_index()
        )
        pivot_df = agg_df.pivot_table(
            index=["task_name", "split"],
            columns="model_name",
            values="score",
            aggfunc="mean",
        ).reset_index()

    elif aggregation_level == "task":
        # For task level, aggregate across both subsets and splits for each task
        agg_df = (
            scores_df.groupby(["model_name", "task_name"])["score"].mean().reset_index()
        )
        pivot_df = agg_df.pivot_table(
            index=["task_name"],
            columns="model_name",
            values="score",
            aggfunc="mean",
        ).reset_index()

    pivot_df.columns.name = None
    model_cols = [
        col for col in pivot_df.columns if col not in ["task_name", "split", "subset"]
    ]
    if model_cols:
        # Create mean row based on aggregation level
        if aggregation_level == "subset":
            # Add an empty row for overall mean
            overall_mean_row = {"task_name": "mean_score", "split": "", "subset": ""}
            for model in model_cols:
                overall_mean_row[model] = pivot_df[model].mean()
            pivot_df = pd.concat(
                [pivot_df, pd.DataFrame([overall_mean_row])], ignore_index=True
            )

        elif aggregation_level == "split":
            overall_mean_row = {"task_name": "mean_score", "split": ""}
            for model in model_cols:
                overall_mean_row[model] = pivot_df[model].mean()
            pivot_df = pd.concat(
                [pivot_df, pd.DataFrame([overall_mean_row])], ignore_index=True
            )

        elif aggregation_level == "task":
            # Add overall mean row
            overall_mean_row = {"task_name": "mean_score"}
            for model in model_cols:
                overall_mean_row[model] = pivot_df[model].mean()
            pivot_df = pd.concat(
                [pivot_df, pd.DataFrame([overall_mean_row])], ignore_index=True
            )

    # Round scores to 2 decimal places
    numeric_columns = pivot_df.select_dtypes(include=np.number).columns
    pivot_df[numeric_columns] = pivot_df[numeric_columns].round(2)

    # Save output if path is provided
    if output_path:
        output_dir = Path(output_path).parent
        os.makedirs(output_dir, exist_ok=True)

        save_dataframe(pivot_df, output_path)
        logger.info(f"Comparison table saved to {output_path}")

    return pivot_df


def format_table_for_display(df: pd.DataFrame) -> str:
    """Format a DataFrame for terminal display."""
    max_rows = 10
    if len(df) > max_rows:
        display_df = df.head(max_rows)
        return f"{display_df.to_string()}\n... {len(df) - max_rows} more rows"
    return df.to_string()


def create_table_cli(args: argparse.Namespace) -> pd.DataFrame:
    """Entry point for CLI integration."""
    models = (
        [model.strip() for model in args.models.split(",")] if args.models else None
    )

    result_df = create_comparison_table(
        results_folder=args.results,
        model_names=models,
        benchmark_name=args.benchmark,
        output_path=args.output,
        aggregation_level=args.aggregation_level,
    )

    # Display table in terminal
    if not result_df.empty:
        print(
            f"\n===== COMPARISON TABLE ({args.aggregation_level.upper()} AGGREGATION) ====="
        )
        print(format_table_for_display(result_df))
    else:
        print("\nNo data available for the specified models and benchmark")

    return result_df
