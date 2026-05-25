"""Plot: model parameter count vs MVEB mean score (efficiency frontier)."""

from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("mteb").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mteb
from scripts.mveb_paper._common import (
    MVEB_BENCHMARK,
    OUT_DIR,
    cache,
    fetch_model_meta,
    load_benchmark_means,
)


def build_df() -> pd.DataFrame:
    means = load_benchmark_means(MVEB_BENCHMARK)
    rows = []
    for model_name in means.index:
        meta = fetch_model_meta(model_name)
        rows.append(
            {
                "model": model_name,
                "mean": means[model_name] * 100,
                "n_parameters": meta["n_parameters"],
            }
        )
    result = pd.DataFrame(rows).dropna(subset=["n_parameters"])
    result["log_params"] = np.log10(result["n_parameters"])
    return result


def plot_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.scatter(
        df["log_params"],
        df["mean"],
        color="#e15759",
        alpha=0.75,
        s=55,
        linewidths=0.3,
        edgecolors="white",
        zorder=3,
    )

    # Pareto frontier (best score achievable at each model size)
    df_sorted = df.sort_values("log_params")
    best_mean = -np.inf
    pareto_rows = []
    for _, row in df_sorted.iterrows():
        if row["mean"] > best_mean:
            best_mean = row["mean"]
            pareto_rows.append(row)
    pareto = pd.DataFrame(pareto_rows)
    ax.plot(
        pareto["log_params"],
        pareto["mean"],
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        zorder=0,
        label="Efficiency frontier",
    )

    for _, row in pareto.iterrows():
        label = row["model"].split("/")[-1]
        if len(label) > 22:
            label = label[:20] + ".."
        ax.annotate(
            label,
            (row["log_params"], row["mean"]),
            fontsize=6.5,
            xytext=(4, 2),
            textcoords="offset points",
            color="#333333",
        )

    xtick_labels = {8: "100M", 9: "1B", 10: "10B"}
    xticks = sorted(xtick_labels)
    ax.set_xticks(xticks)
    ax.set_xticklabels([xtick_labels[x] for x in xticks])
    ax.set_xlim(8, 10)
    ax.set_xlabel("Number of Parameters", fontsize=10)
    ax.set_ylabel("MVEB Mean Score (%)", fontsize=10)
    ax.set_title("Model Size vs. MVEB Performance", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    print(f"mteb version: {mteb.__version__}")

    print("Loading MVEB results...")
    df = build_df()
    print(f"  {len(df)} models with parameter info")
    if df.empty:
        print("No models with parameter metadata found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_panel(ax, df)
    fig.tight_layout()

    out_pdf = OUT_DIR / "plot_size_vs_performance.pdf"
    out_png = OUT_DIR / "plot_size_vs_performance.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_pdf}")

    print("\nTop models by mean score:")
    for _, row in df.nlargest(10, "mean").iterrows():
        print(f"  {row['mean']:5.1f}%  {int(row['n_parameters']):>12,}  {row['model']}")


if __name__ == "__main__":
    main()
