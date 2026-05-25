"""Analyze task-to-task and type-to-type correlations for MVEB.

Outputs:
  - mveb_task_correlation.csv         Spearman correlation matrix (tasks × tasks)
  - mveb_task_correlation_plot.pdf/png  Heatmap figure
  - mveb_task_correlation.tex         LaTeX heatmap table
  - mveb_type_correlation.tex         LaTeX table of type-subset vs full-benchmark correlation
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("mteb").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mteb
from scripts.mveb_paper._common import (
    MVEB_BENCHMARK,
    OUT_DIR,
    TYPE_ABBREV,
    TYPE_COLORS,
    TYPE_ORDER,
    cache,
)

HIGH_CORR_THRESHOLD = 0.90
MIN_MODELS_FOR_CORR = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_score_matrix(tasks: list) -> pd.DataFrame:
    """Return DataFrame rows=tasks, cols=models (0–1 scale)."""
    results = cache.load_results(tasks=tasks, require_model_meta=False)
    return results.to_dataframe().set_index("task_name")


def build_model_task_matrix(
    score_df: pd.DataFrame, task_names: list[str]
) -> pd.DataFrame:
    """Return (models × tasks) matrix, dropping models with all-NaN rows."""
    available = [t for t in task_names if t in score_df.index]
    return score_df.loc[available].T.dropna(how="all")


# ---------------------------------------------------------------------------
# Pairwise task correlation
# ---------------------------------------------------------------------------


def pairwise_spearman(mt: pd.DataFrame) -> pd.DataFrame:
    tasks = mt.columns.tolist()
    n = len(tasks)
    corr_vals = np.full((n, n), np.nan)

    for i, t1 in enumerate(tasks):
        corr_vals[i, i] = 1.0
        for j, t2 in enumerate(tasks):
            if j <= i:
                continue
            pair = mt[[t1, t2]].dropna()
            if len(pair) < MIN_MODELS_FOR_CORR:
                continue
            r, _ = spearmanr(pair[t1], pair[t2])
            corr_vals[i, j] = r
            corr_vals[j, i] = r

    return pd.DataFrame(corr_vals, index=tasks, columns=tasks)


def get_high_corr_pairs(
    corr: pd.DataFrame, threshold: float
) -> list[tuple[str, str, float]]:
    pairs = []
    tasks = corr.columns.tolist()
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i + 1 :]:
            v = corr.loc[t1, t2]
            if not np.isnan(v) and v >= threshold:
                pairs.append((t1, t2, float(v)))
    return sorted(pairs, key=lambda x: -x[2])


# ---------------------------------------------------------------------------
# Type-subset vs full-benchmark correlation
# ---------------------------------------------------------------------------


def type_vs_full_correlations(
    mt: pd.DataFrame,
    task_names: list[str],
    task_type_map: dict[str, str],
) -> pd.DataFrame:
    groups: dict[str, list[str]] = defaultdict(list)
    for t in task_names:
        if t in mt.columns:
            groups[task_type_map[t]].append(t)

    available = [t for t in task_names if t in mt.columns]
    full_mean = mt[available].mean(axis=1, skipna=True)

    rows = []
    for task_type in TYPE_ORDER:
        if task_type not in groups:
            continue
        tasks_in_type = groups[task_type]
        subset_mean = mt[tasks_in_type].mean(axis=1, skipna=True)
        valid = full_mean.notna() & subset_mean.notna()
        n_valid = int(valid.sum())
        r = (
            float(spearmanr(full_mean[valid], subset_mean[valid])[0])
            if n_valid >= MIN_MODELS_FOR_CORR
            else np.nan
        )
        rows.append(
            {
                "task_type": task_type,
                "abbrev": TYPE_ABBREV.get(task_type, task_type),
                "n_tasks": len(tasks_in_type),
                "n_models": n_valid,
                "spearman_r": r,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------


def coverage_summary(tasks: list) -> None:
    domains: set[str] = set()
    type_counts: Counter = Counter()
    modalities: set[str] = set()
    for task in tasks:
        meta = task.metadata
        domains.update(meta.domains or [])
        type_counts[meta.type] += 1
        modalities.update(getattr(meta, "modalities", None) or [])

    print(f"\nCoverage over {len(tasks)} tasks:")
    print(f"  Domains   : {len(domains)} — {', '.join(sorted(domains))}")
    print(f"  Modalities: {', '.join(sorted(modalities))}")
    print(f"  Task types: {dict(type_counts.most_common())}")


# ---------------------------------------------------------------------------
# LaTeX output helpers
# ---------------------------------------------------------------------------


def _color_cell(v: float) -> str:
    if np.isnan(v):
        return "\\cellcolor{white!0}--"
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        pct = int(round(v * 40))
        color = f"blue!{pct}"
    else:
        pct = int(round(-v * 40))
        color = f"red!{pct}"
    return f"\\cellcolor{{{color}}}{v:.2f}"


def _sorted_tasks(corr: pd.DataFrame, task_type_map: dict[str, str]) -> list[str]:
    type_order_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    return sorted(
        corr.columns.tolist(),
        key=lambda t: (type_order_idx.get(task_type_map.get(t, ""), 99), t),
    )


def generate_corr_heatmap_latex(
    corr: pd.DataFrame, task_type_map: dict[str, str]
) -> str:
    tasks = _sorted_tasks(corr, task_type_map)
    corr = corr.loc[tasks, tasks]
    n = len(tasks)

    short = [t[:20] + ".." if len(t) > 22 else t for t in tasks]
    short = [s.replace("_", "\\_") for s in short]

    col_spec = "l" + "c" * n
    header = " & ".join(f"\\rotatebox{{90}}{{\\small {s}}}" for s in short)

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Pairwise Spearman correlation between MVEB tasks "
        "(computed over model rankings). Blue = positive, red = negative correlation.}",
        "    \\resizebox{\\linewidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Task}} & {header} \\\\",
        "    \\midrule",
    ]

    current_type = None
    for task, s in zip(tasks, short):
        t = task_type_map.get(task, "")
        if t != current_type:
            current_type = t
            lines.append(
                f"    \\multicolumn{{{n + 1}}}{{l}}{{\\textit{{{TYPE_ABBREV.get(t, t)}}}}} \\\\"
            )
        cells = " & ".join(_color_cell(corr.loc[task, t2]) for t2 in tasks)
        lines.append(f"    {s} & {cells} \\\\")

    lines += [
        "    \\bottomrule",
        f"    \\end{{tabular}}",
        "    }",
        "    \\label{tab:mveb_task_corr}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def generate_type_corr_latex(type_df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Spearman correlation between each task-type subset mean "
        "and the full MVEB mean across all evaluated models.}",
        "    \\begin{tabular}{lccc}",
        "    \\toprule",
        "    \\textbf{Task Type} & \\textbf{N Tasks} & \\textbf{N Models} & \\textbf{Spearman $r$} \\\\",
        "    \\midrule",
    ]
    for _, row in type_df.iterrows():
        r_str = f"{row['spearman_r']:.3f}" if not np.isnan(row["spearman_r"]) else "--"
        lines.append(
            f"    {row['abbrev']} & {int(row['n_tasks'])} & {int(row['n_models'])} & {r_str} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    \\label{tab:mveb_type_corr}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------


def plot_task_correlation_heatmap(
    corr: pd.DataFrame,
    task_type_map: dict[str, str],
    out_path: Path,
) -> None:
    tasks = _sorted_tasks(corr, task_type_map)
    corr_sorted = corr.loc[tasks, tasks].values.astype(float)
    n = len(tasks)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), max(8, n * 0.42)))
    im = ax.imshow(corr_sorted, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman $r$")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    prev_type = None
    boundaries = []
    tick_colors = []
    for idx, task in enumerate(tasks):
        ttype = task_type_map.get(task, "")
        if ttype != prev_type:
            if idx > 0:
                boundaries.append(idx - 0.5)
            prev_type = ttype
        tick_colors.append(TYPE_COLORS.get(ttype, "black"))

    xlabels = ax.set_xticklabels(tasks, rotation=90, fontsize=7, ha="right")
    ylabels = ax.set_yticklabels(tasks, fontsize=7)
    for lbl, color in zip(xlabels, tick_colors):
        lbl.set_color(color)
    for lbl, color in zip(ylabels, tick_colors):
        lbl.set_color(color)

    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(b, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    present_types = list(dict.fromkeys(task_type_map.get(t, "") for t in tasks))
    legend_elements = [
        mpatches.Patch(
            facecolor=TYPE_COLORS.get(tt, "gray"), label=TYPE_ABBREV.get(tt, tt)
        )
        for tt in present_types
        if tt
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.22, 1.0),
        fontsize=8,
        title="Task type",
        title_fontsize=8,
    )

    ax.set_title("Pairwise Spearman Correlation Between MVEB Tasks", fontsize=11)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"mteb version: {mteb.__version__}")
    print(f"\nLoading {MVEB_BENCHMARK}...")
    bench = mteb.get_benchmark(MVEB_BENCHMARK)
    tasks = bench.tasks
    task_names = [t.metadata.name for t in tasks]
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}

    coverage_summary(tasks)

    print("\nLoading scores from cache...")
    score_df = load_score_matrix(tasks)
    mt = build_model_task_matrix(score_df, task_names)
    print(f"  {mt.shape[0]} models × {mt.shape[1]} tasks")

    # Task-task correlation
    print("\nComputing pairwise task correlations...")
    corr = pairwise_spearman(mt)

    corr_csv = OUT_DIR / "mveb_task_correlation.csv"
    corr.to_csv(corr_csv)
    print(f"Saved correlation matrix to {corr_csv}")

    high_pairs = get_high_corr_pairs(corr, HIGH_CORR_THRESHOLD)
    print(f"\nHighly correlated pairs (r >= {HIGH_CORR_THRESHOLD}):")
    if high_pairs:
        for t1, t2, r in high_pairs:
            type1 = TYPE_ABBREV.get(task_type_map.get(t1, ""), "?")
            type2 = TYPE_ABBREV.get(task_type_map.get(t2, ""), "?")
            print(f"  {t1} [{type1}]  ↔  {t2} [{type2}]  r={r:.3f}")
    else:
        print("  (none above threshold)")

    print("\nMean pairwise |correlation| per task (potential redundancy):")
    mean_corr = corr.abs().apply(lambda col: col[col.index != col.name].mean(), axis=0)
    for task in mean_corr.sort_values(ascending=False).head(10).index:
        print(f"  {task}: avg |r| = {mean_corr[task]:.3f}")

    corr_tex = OUT_DIR / "mveb_task_correlation.tex"
    corr_tex.write_text(generate_corr_heatmap_latex(corr, task_type_map))
    print(f"\nCorrelation heatmap table written to {corr_tex}")

    print("\nPlotting task correlation heatmap...")
    plot_task_correlation_heatmap(
        corr,
        task_type_map,
        OUT_DIR / "mveb_task_correlation_plot",
    )

    # Type-subset vs full-benchmark correlation
    print("\nComputing type-subset vs full-benchmark correlations...")
    type_df = type_vs_full_correlations(mt, task_names, task_type_map)
    print(
        type_df[["abbrev", "n_tasks", "n_models", "spearman_r"]].to_string(index=False)
    )

    type_tex = OUT_DIR / "mveb_type_correlation.tex"
    type_tex.write_text(generate_type_corr_latex(type_df))
    print(f"Type correlation table written to {type_tex}")


if __name__ == "__main__":
    main()
