"""
MAEB Lite vs Extended Benchmark Correlation Analysis

This script computes the correlation between lite and extended benchmark variants
to validate that lite benchmarks preserve model rankings while reducing evaluation time.

We analyze two benchmark pairs:
1. Audio-only: MAEB(audio, lite) vs MAEB(audio, extended)
2. Audio-text: MAEB(audio-text, lite) vs MAEB(audio-text, extended)
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
    audio_lite = get_benchmark("MAEB(audio, lite)")
    audio_extended = get_benchmark("MAEB(audio, extended)")
    audio_text_lite = get_benchmark("MAEB(audio-text, lite)")
    audio_text_extended = get_benchmark("MAEB(audio-text, extended)")

    # Get task names from each benchmark
    audio_lite_tasks = [t.metadata.name for t in audio_lite.tasks]
    audio_extended_tasks = [t.metadata.name for t in audio_extended.tasks]
    audio_text_lite_tasks = [t.metadata.name for t in audio_text_lite.tasks]
    audio_text_extended_tasks = [t.metadata.name for t in audio_text_extended.tasks]

    print(f"\nAudio-only lite tasks: {len(audio_lite_tasks)}")
    print(f"Audio-only extended tasks: {len(audio_extended_tasks)}")
    print(f"Audio-text lite tasks: {len(audio_text_lite_tasks)}")
    print(f"Audio-text extended tasks: {len(audio_text_extended_tasks)}")

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
    all_task_names = list(
        set(audio_lite_tasks)
        | set(audio_extended_tasks)
        | set(audio_text_lite_tasks)
        | set(audio_text_extended_tasks)
    )
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

    # === Audio-Only Benchmark Correlation ===
    print("\n" + "=" * 50)
    print("Audio-Only Benchmark Correlation")
    print("=" * 50)

    # Filter tasks to those present in results
    audio_lite_available = [t for t in audio_lite_tasks if t in results_df.columns]
    audio_extended_available = [
        t for t in audio_extended_tasks if t in results_df.columns
    ]

    print(
        f"Audio lite tasks available: {len(audio_lite_available)}/{len(audio_lite_tasks)}"
    )
    print(
        f"Audio extended tasks available: {len(audio_extended_available)}/{len(audio_extended_tasks)}"
    )

    # Filter to models with complete results on BOTH benchmarks
    audio_complete_mask = results_df[audio_lite_available].notna().all(
        axis=1
    ) & results_df[audio_extended_available].notna().all(axis=1)
    audio_filtered_df = results_df[audio_complete_mask]
    print(f"Models with complete audio results: {len(audio_filtered_df)}")

    # Compute average scores
    audio_avg_lite = audio_filtered_df[audio_lite_available].mean(axis=1)
    audio_avg_extended = audio_filtered_df[audio_extended_available].mean(axis=1)

    # Compute correlations
    audio_spearman, audio_spearman_p = spearmanr(audio_avg_lite, audio_avg_extended)
    audio_pearson, audio_pearson_p = pearsonr(audio_avg_lite, audio_avg_extended)

    print(f"\nSpearman correlation: {audio_spearman:.4f} (p={audio_spearman_p:.2e})")
    print(f"Pearson correlation: {audio_pearson:.4f} (p={audio_pearson_p:.2e})")

    # === Audio-Text Benchmark Correlation ===
    print("\n" + "=" * 50)
    print("Audio-Text Benchmark Correlation")
    print("=" * 50)

    # Filter tasks to those present in results
    audio_text_lite_available = [
        t for t in audio_text_lite_tasks if t in results_df.columns
    ]
    audio_text_extended_available = [
        t for t in audio_text_extended_tasks if t in results_df.columns
    ]

    print(
        f"Audio-text lite tasks available: {len(audio_text_lite_available)}/{len(audio_text_lite_tasks)}"
    )
    print(
        f"Audio-text extended tasks available: {len(audio_text_extended_available)}/{len(audio_text_extended_tasks)}"
    )

    # Filter to models with complete results on BOTH benchmarks
    audio_text_complete_mask = results_df[audio_text_lite_available].notna().all(
        axis=1
    ) & results_df[audio_text_extended_available].notna().all(axis=1)
    audio_text_filtered_df = results_df[audio_text_complete_mask]
    print(f"Models with complete audio-text results: {len(audio_text_filtered_df)}")

    # Compute average scores
    audio_text_avg_lite = audio_text_filtered_df[audio_text_lite_available].mean(axis=1)
    audio_text_avg_extended = audio_text_filtered_df[
        audio_text_extended_available
    ].mean(axis=1)

    # Compute correlations
    audio_text_spearman, audio_text_spearman_p = spearmanr(
        audio_text_avg_lite, audio_text_avg_extended
    )
    audio_text_pearson, audio_text_pearson_p = pearsonr(
        audio_text_avg_lite, audio_text_avg_extended
    )

    print(
        f"\nSpearman correlation: {audio_text_spearman:.4f} (p={audio_text_spearman_p:.2e})"
    )
    print(
        f"Pearson correlation: {audio_text_pearson:.4f} (p={audio_text_pearson_p:.2e})"
    )

    # === Summary ===
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    summary_data = {
        "Benchmark": ["Audio-only", "Audio-text"],
        "Models": [len(audio_filtered_df), len(audio_text_filtered_df)],
        "Spearman": [audio_spearman, audio_text_spearman],
        "Pearson": [audio_pearson, audio_text_pearson],
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n=== For exp.tex ===")
    print(
        f"Spearman rho={audio_spearman:.2f} for audio-only and rho={audio_text_spearman:.2f} for audio-text"
    )

    # === Create scatter plots ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Audio-only plot
    ax1 = axes[0]
    ax1.scatter(audio_avg_lite, audio_avg_extended, alpha=0.7, s=60)
    z1 = np.polyfit(audio_avg_lite, audio_avg_extended, 1)
    p1 = np.poly1d(z1)
    x_line1 = np.linspace(audio_avg_lite.min(), audio_avg_lite.max(), 100)
    ax1.plot(x_line1, p1(x_line1), "r--", alpha=0.8, label="Linear fit")
    min_val1 = min(audio_avg_lite.min(), audio_avg_extended.min())
    max_val1 = max(audio_avg_lite.max(), audio_avg_extended.max())
    ax1.plot([min_val1, max_val1], [min_val1, max_val1], "k:", alpha=0.5, label="y=x")
    ax1.set_xlabel("Average Score (Lite)", fontsize=12)
    ax1.set_ylabel("Average Score (Extended)", fontsize=12)
    ax1.set_title(
        f"Audio-Only: Lite vs Extended\nSpearman={audio_spearman:.3f}, Pearson={audio_pearson:.3f}",
        fontsize=14,
    )
    ax1.legend()

    # Audio-text plot
    ax2 = axes[1]
    ax2.scatter(audio_text_avg_lite, audio_text_avg_extended, alpha=0.7, s=60)
    z2 = np.polyfit(audio_text_avg_lite, audio_text_avg_extended, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(audio_text_avg_lite.min(), audio_text_avg_lite.max(), 100)
    ax2.plot(x_line2, p2(x_line2), "r--", alpha=0.8, label="Linear fit")
    min_val2 = min(audio_text_avg_lite.min(), audio_text_avg_extended.min())
    max_val2 = max(audio_text_avg_lite.max(), audio_text_avg_extended.max())
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], "k:", alpha=0.5, label="y=x")
    ax2.set_xlabel("Average Score (Lite)", fontsize=12)
    ax2.set_ylabel("Average Score (Extended)", fontsize=12)
    ax2.set_title(
        f"Audio-Text: Lite vs Extended\nSpearman={audio_text_spearman:.3f}, Pearson={audio_text_pearson:.3f}",
        fontsize=14,
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("benchmark_lite_vs_extended_correlation.png", dpi=150)
    plt.savefig("benchmark_lite_vs_extended_correlation.pdf")
    print("\nPlot saved to: benchmark_lite_vs_extended_correlation.png")
    print("Plot saved to: benchmark_lite_vs_extended_correlation.pdf")


if __name__ == "__main__":
    main()
