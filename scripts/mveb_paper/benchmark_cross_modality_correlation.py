"""MVEB vs MTEB(eng) / MIEB(Img) / MAEB(beta) cross-modality scatter.

For each model that has results on both MVEB and a counterpart benchmark,
plot MVEB score (y-axis) against the counterpart score (x-axis).
Every point is labelled.  Produces a 1×3 figure plus a LaTeX table.
"""

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
from scipy.stats import pearsonr, spearmanr

import mteb
from scripts.mveb_paper._common import (
    COUNTERPART_COLORS,
    MVEB_BENCHMARK,
    OUT_DIR,
    cache,
    load_benchmark_means,
    short_name,
)

MIN_OVERLAP = 2


def scatter_panel(
    ax: plt.Axes,
    mveb_scores,
    other_scores,
    other_label: str,
    color: str,
) -> dict | None:
    common = mveb_scores.index.intersection(other_scores.index)
    valid = mveb_scores[common].notna() & other_scores[common].notna()
    xv = other_scores[common][valid]
    yv = mveb_scores[common][valid]

    print(f"  {other_label}: {len(xv)} models in common")
    if len(xv) < MIN_OVERLAP:
        ax.text(
            0.5,
            0.5,
            f"Insufficient overlap\n(n={len(xv)})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        ax.set_title(f"MVEB vs {other_label}", fontsize=11)
        return None

    spearman_r, spearman_p = spearmanr(xv, yv)
    pearson_r, pearson_p = pearsonr(xv, yv)

    ax.scatter(
        xv * 100,
        yv * 100,
        color=color,
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    for model in xv.index:
        ax.annotate(
            short_name(model),
            (xv[model] * 100, yv[model] * 100),
            fontsize=7,
            xytext=(5, 3),
            textcoords="offset points",
            color="#333333",
        )

    if len(xv) >= 2:
        m, b = np.polyfit(xv.values, yv.values, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(
            xs * 100, (m * xs + b) * 100, "--", color="black", linewidth=1.0, alpha=0.5
        )

    ax.set_xlabel(f"{other_label} Mean (%)", fontsize=10)
    ax.set_ylabel("MVEB Mean (%)", fontsize=10)
    ax.set_title(
        f"MVEB vs {other_label}\nSpearman ρ = {spearman_r:.3f}, Pearson r = {pearson_r:.3f}  (n={len(xv)})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return {
        "comparison": f"MVEB vs {other_label}",
        "n": len(xv),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


def main() -> None:
    print(f"mteb version: {mteb.__version__}")

    print("\nLoading MVEB scores...", end=" ", flush=True)
    mveb_scores = load_benchmark_means(MVEB_BENCHMARK, require_all_tasks=True)
    print(f"{len(mveb_scores)} models")

    print("Loading counterpart benchmark scores:")
    counterpart_scores = {}
    for bench_name in COUNTERPART_COLORS:
        print(f"  {bench_name}...", end=" ", flush=True)
        s = load_benchmark_means(bench_name)
        counterpart_scores[bench_name] = s
        print(f"{len(s)} models")

    n_panels = len(COUNTERPART_COLORS)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    results = []
    print("\nBuilding scatter panels:")
    for ax, (bench_name, color) in zip(axes, COUNTERPART_COLORS.items()):
        row = scatter_panel(
            ax, mveb_scores, counterpart_scores[bench_name], bench_name, color
        )
        if row:
            results.append(row)
            print(
                f"    Spearman ρ = {row['spearman_r']:.4f}  (p={row['spearman_p']:.2e})"
            )
            print(
                f"    Pearson  r = {row['pearson_r']:.4f}  (p={row['pearson_p']:.2e})"
            )

    fig.suptitle("MVEB vs Text, Image & Audio Benchmarks", fontsize=13, y=1.02)
    fig.tight_layout()

    out_png = OUT_DIR / "benchmark_cross_modality_correlation.png"
    out_pdf = OUT_DIR / "benchmark_cross_modality_correlation.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nPlot saved to {out_png}")
    plt.close(fig)

    if not results:
        return

    tex_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Rank correlation between MVEB and general-purpose benchmarks. "
        "Low correlation indicates that video embedding evaluation captures distinct capabilities.}",
        "    \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\textbf{Comparison} & \\textbf{N} "
        "& \\textbf{Spearman $\\rho$} & \\textbf{Pearson $r$} & \\textbf{p-value} \\\\",
        "    \\midrule",
    ]
    for row in results:
        tex_lines.append(
            f"    {row['comparison']} & {row['n']} "
            f"& {row['spearman_r']:.3f} & {row['pearson_r']:.3f} "
            f"& {row['spearman_p']:.2e} \\\\"
        )
    tex_lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    \\label{tab:cross_modality_corr}",
        "\\end{table}",
    ]
    tex_out = OUT_DIR / "benchmark_cross_modality_correlation.tex"
    tex_out.write_text("\n".join(tex_lines))
    print(f"LaTeX table written to {tex_out}")


if __name__ == "__main__":
    main()
