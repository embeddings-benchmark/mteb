#!/usr/bin/env python3
"""
Calculate the correlation between MAEB and MAEB(extended) benchmark scores.

This script computes the Spearman and Pearson correlations between the average
model performance on MAEB vs MAEB(extended) to validate that the smaller
benchmark preserves model rankings.
"""

from pathlib import Path

from scipy.stats import pearsonr, spearmanr

import mteb
from mteb.cache import ResultCache


def main():
    # Load results from cache
    results_dir = Path("/Users/isaac/work/maeb-results/results")
    model_names = [
        folder.name.replace("__", "/")
        for folder in results_dir.iterdir()
        if folder.is_dir()
    ]
    print(f"Found {len(model_names)} models")

    # Get model metadata
    models = [mteb.get_model_meta(name) for name in model_names]

    # Load benchmarks
    maeb = mteb.get_benchmark("MAEB")
    maeb_extended = mteb.get_benchmark("MAEB(extended)")

    print(f"MAEB tasks: {len(maeb.tasks)}")
    print(f"MAEB(extended) tasks: {len(maeb_extended.tasks)}")

    # Load results
    cache = ResultCache(cache_path="/Users/isaac/work/maeb-results")

    # Get results for MAEB(extended) - the superset
    mteb_results = cache.load_results(
        models=models, tasks=maeb_extended.tasks, require_model_meta=False
    )

    # Create dataframe with all results
    full_df = mteb_results.to_dataframe().set_index("task_name").T

    # Get task names for each benchmark
    maeb_task_names = [t.metadata.name for t in maeb.tasks]
    maeb_extended_task_names = [t.metadata.name for t in maeb_extended.tasks]

    # Filter to tasks that have results
    maeb_tasks_with_results = [t for t in maeb_task_names if t in full_df.columns]
    maeb_extended_tasks_with_results = [
        t for t in maeb_extended_task_names if t in full_df.columns
    ]

    print(
        f"\nMAEB tasks with results: {len(maeb_tasks_with_results)}/{len(maeb_task_names)}"
    )
    print(
        f"MAEB(extended) tasks with results: {len(maeb_extended_tasks_with_results)}/{len(maeb_extended_task_names)}"
    )

    # Filter models with sufficient results (drop models with too many NaNs)
    maeb_df = full_df[maeb_tasks_with_results]
    maeb_extended_df = full_df[maeb_extended_tasks_with_results]

    # Keep models with at most 10 NaN values in each benchmark
    maeb_nan_counts = maeb_df.isna().sum(axis=1)
    maeb_extended_nan_counts = maeb_extended_df.isna().sum(axis=1)

    valid_models = (maeb_nan_counts <= 10) & (maeb_extended_nan_counts <= 10)
    maeb_df = maeb_df[valid_models]
    maeb_extended_df = maeb_extended_df[valid_models]

    print(f"\nModels with sufficient results: {len(maeb_df)}")

    # Compute average performance per model for each benchmark
    maeb_avg = maeb_df.mean(axis=1)
    maeb_extended_avg = maeb_extended_df.mean(axis=1)

    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(maeb_avg, maeb_extended_avg)
    pearson_corr, pearson_p = pearsonr(maeb_avg, maeb_extended_avg)

    print("\n" + "=" * 60)
    print("MAEB vs MAEB(extended) Correlation Analysis")
    print("=" * 60)
    print(f"\nSpearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.2e})")

    # Show some statistics
    print(f"\nMAEB average score range: [{maeb_avg.min():.3f}, {maeb_avg.max():.3f}]")
    print(
        f"MAEB(extended) average score range: [{maeb_extended_avg.min():.3f}, {maeb_extended_avg.max():.3f}]"
    )

    # Show top 5 models by each benchmark
    print("\nTop 5 models by MAEB:")
    for i, (model, score) in enumerate(
        maeb_avg.sort_values(ascending=False).head().items(), 1
    ):
        ext_score = maeb_extended_avg[model]
        print(f"  {i}. {model}: {score:.3f} (extended: {ext_score:.3f})")

    print("\nTop 5 models by MAEB(extended):")
    for i, (model, score) in enumerate(
        maeb_extended_avg.sort_values(ascending=False).head().items(), 1
    ):
        maeb_score = maeb_avg[model]
        print(f"  {i}. {model}: {score:.3f} (MAEB: {maeb_score:.3f})")


if __name__ == "__main__":
    main()
