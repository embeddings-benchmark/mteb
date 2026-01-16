"""
MAEB vs MAEB(extended) Benchmark Correlation Analysis

This script computes the correlation between MAEB (lite) and MAEB(extended) benchmark variants
to validate that the lite benchmark preserves model rankings while reducing evaluation time.

MAEB is the main benchmark (35 tasks) while MAEB(extended) is an intermediate filtering step
containing 91 tasks that combines audio-only and audio-text evaluation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import mteb
from mteb.benchmarks import get_benchmark
from mteb.cache import ResultCache


def main():
    print(f"MTEB version: {mteb.__version__}")

    # Get benchmarks by name
    maeb_lite = get_benchmark("MAEB")
    maeb_extended = get_benchmark("MAEB(extended)")

    # Get task names from each benchmark
    lite_tasks = [t.metadata.name for t in maeb_lite.tasks]
    extended_tasks = [t.metadata.name for t in maeb_extended.tasks]

    print(f"\nMAEB (lite) tasks: {len(lite_tasks)}")
    print(f"MAEB (extended) tasks: {len(extended_tasks)}")

    # Load model names from results directory
    results_dir = Path("/Users/isaac/work/maeb-results/results")
    model_names = [
        folder.name.replace("__", "/")
        for folder in results_dir.iterdir()
        if folder.is_dir()
    ]
    print(f"\nTotal models found: {len(model_names)}")

    # Get model metadata
    models: list[mteb.ModelMeta] = [mteb.get_model_meta(name) for name in model_names]

    # Get missing revisions
    for model in models:
        if model.revision is None:
            print(f"Getting revision for {model.name}")
            encoder = model.load_model()
            model.revision = encoder.model_card_data.base_model_revision  # type: ignore

    # Combine all tasks for loading
    all_task_names = list(set(lite_tasks) | set(extended_tasks))
    all_tasks = mteb.get_tasks(tasks=all_task_names)

    # Load results
    cache = ResultCache(cache_path="/Users/isaac/work/maeb-results")
    mteb_results = cache.load_results(
        models=models, tasks=all_tasks, require_model_meta=False
    )
    print(f"Loaded results for {len(mteb_results.model_results)} models")

    # Create full results dataframe (rows=models, cols=tasks)
    results_df = mteb_results.to_dataframe().set_index("task_name").T
    print(f"Results DataFrame shape: {results_df.shape}")

    # === MAEB vs MAEB(extended) Benchmark Correlation ===
    print("\n" + "=" * 50)
    print("MAEB vs MAEB(extended) Benchmark Correlation")
    print("=" * 50)

    # Filter tasks to those present in results
    lite_available = [t for t in lite_tasks if t in results_df.columns]
    extended_available = [t for t in extended_tasks if t in results_df.columns]

    print(f"MAEB (lite) tasks available: {len(lite_available)}/{len(lite_tasks)}")
    print(
        f"MAEB (extended) tasks available: {len(extended_available)}/{len(extended_tasks)}"
    )

    # Filter to models with complete results on BOTH benchmarks
    complete_mask = results_df[lite_available].notna().all(axis=1) & results_df[
        extended_available
    ].notna().all(axis=1)
    filtered_df = results_df[complete_mask]
    print(f"Models with complete results on both benchmarks: {len(filtered_df)}")

    # Compute average scores
    avg_lite = filtered_df[lite_available].mean(axis=1)
    avg_extended = filtered_df[extended_available].mean(axis=1)

    # Compute correlations
    spearman_corr, spearman_p = spearmanr(avg_lite, avg_extended)
    pearson_corr, pearson_p = pearsonr(avg_lite, avg_extended)

    print(f"\nSpearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.2e})")

    # === Summary ===
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    summary_data = {
        "Benchmark Pair": ["MAEB vs MAEB(extended)"],
        "Models": [len(filtered_df)],
        "Spearman": [spearman_corr],
        "Pearson": [pearson_corr],
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n=== For method.tex ===")
    print(f"Spearman rho={spearman_corr:.2f} for MAEB vs MAEB(extended)")

    # === Create scatter plot ===
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(avg_lite, avg_extended, alpha=0.7, s=60)
    z = np.polyfit(avg_lite, avg_extended, 1)
    p = np.poly1d(z)
    x_line = np.linspace(avg_lite.min(), avg_lite.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")
    min_val = min(avg_lite.min(), avg_extended.min())
    max_val = max(avg_lite.max(), avg_extended.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k:", alpha=0.5, label="y=x")
    ax.set_xlabel("Average Score (MAEB)", fontsize=12)
    ax.set_ylabel("Average Score (MAEB Extended)", fontsize=12)
    ax.set_title(
        f"MAEB vs MAEB(extended)\nSpearman={spearman_corr:.3f}, Pearson={pearson_corr:.3f}",
        fontsize=14,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig("benchmark_lite_vs_extended_correlation.png", dpi=150)
    plt.savefig("benchmark_lite_vs_extended_correlation.pdf")
    print("\nPlot saved to: benchmark_lite_vs_extended_correlation.png")
    print("Plot saved to: benchmark_lite_vs_extended_correlation.pdf")


if __name__ == "__main__":
    main()
