from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

import mteb
from mteb.leaderboard.table import create_tables
from mteb.load_results import load_results

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def get_available_benchmarks():
    """Get all available benchmark names."""
    return [b.name for b in mteb.get_benchmarks()]


def load_leaderboard(
    benchmark_name: str,
    results_repo: str,
    models: list[str] | None = None,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MTEB leaderboard for a specific benchmark, results repo, and models.

    Args:
        benchmark_name: Name of the benchmark (e.g., "MTEB(eng)").
        models: List of model names to include. Default is None (all models).
        results_repo: Path to results repo (local or remote). Default is None (default repo).
        save_path: Path to save the leaderboard as CSV.

    Returns:
        summary_df: Leaderboard summary table.
        per_task_df: Per-task leaderboard table.
    """
    logger.info(f"Loading benchmark: {benchmark_name}")

    # Get the selected benchmark
    benchmarks = mteb.get_benchmarks()
    benchmark = next((b for b in benchmarks if b.name == benchmark_name), None)
    if not benchmark:
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found. Available: {get_available_benchmarks()}"
        )

    # Load all results from the specified repository
    benchmark_results = load_results(
        results_repo=results_repo,
        only_main_score=True,
        require_model_meta=False,
    )

    # Filter results for the selected benchmark
    benchmark_results_filtered = benchmark.load_results(
        base_results=benchmark_results
    ).join_revisions()

    # Convert scores into long format
    scores_long = benchmark_results_filtered.get_scores(format="long")

    # Convert scores into leaderboard tables
    summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)

    # Convert Gradio DataFrames to Pandas
    summary_df = pd.DataFrame(
        summary_gr_df.value["data"], columns=summary_gr_df.value["headers"]
    )
    per_task_df = pd.DataFrame(
        per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"]
    )

    # Filter models if specified
    if models:
        summary_df = summary_df[
            summary_df["Model"].apply(lambda x: any(m in x for m in models))
        ]
        per_task_df = per_task_df[
            per_task_df["Model"].apply(lambda x: any(m in x for m in models))
        ]

    # Save to CSV if save_path is specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

        summary_file = os.path.join(
            save_path, f"{benchmark_name.replace(' ', '_')}_summary.csv"
        )
        per_task_file = os.path.join(
            save_path, f"{benchmark_name.replace(' ', '_')}_per_task.csv"
        )

        summary_df.to_csv(summary_file, index=False)
        per_task_df.to_csv(per_task_file, index=False)

        logger.info(f"Leaderboard saved to {summary_file}")
        logger.info(f"Per-task results saved to {per_task_file}")

    return summary_df, per_task_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save an MTEB leaderboard for a specified benchmark."
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help=f"Which benchmark to load. Available: {get_available_benchmarks()}",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="List of model names to include (default: all models).",
    )

    parser.add_argument(
        "--results_repo",
        type=str,
        default="https://github.com/embeddings-benchmark/results",
        help="Path to results repository. Default is the official MTEB results repo.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the leaderboard as a CSV file",
    )

    args = parser.parse_args()
    summary_df, per_task_df = load_leaderboard(
        benchmark_name=args.benchmark,
        models=args.models,
        results_repo=args.results_repo,
        save_path=args.save_path,
    )

    print("\n===== SUMMARY TABLE =====")
    print(summary_df)

    print("\n===== PER-TASK TABLE =====")
    print(per_task_df)
