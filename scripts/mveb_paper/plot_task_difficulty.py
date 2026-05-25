"""Plot: MVEB task difficulty (mean score per task across all evaluated models)."""

from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("mteb").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import mteb
from scripts.mveb_paper._common import (
    MVEB_BENCHMARK,
    OUT_DIR,
    TYPE_ABBREV,
    TYPE_COLORS,
    cache,
)


def main() -> None:
    print(f"mteb version: {mteb.__version__}")

    bench = mteb.get_benchmark(MVEB_BENCHMARK)
    tasks = bench.tasks
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}

    results = cache.load_results(tasks=tasks, require_model_meta=False)
    score_df = results.to_dataframe().set_index("task_name")

    task_means: dict[str, float] = {}
    task_stds: dict[str, float] = {}
    task_n: dict[str, int] = {}
    for task in task_type_map:
        if task in score_df.index:
            vals = score_df.loc[task].dropna().astype(float) * 100
            if len(vals) > 0:
                task_means[task] = float(vals.mean())
                task_stds[task] = float(vals.std())
                task_n[task] = len(vals)

    sorted_tasks = sorted(task_means, key=task_means.get)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_tasks) * 0.42)))

    y_pos = np.arange(len(sorted_tasks))
    means = [task_means[t] for t in sorted_tasks]
    stds = [task_stds.get(t, 0.0) for t in sorted_tasks]
    colors = [
        TYPE_COLORS.get(task_type_map.get(t, ""), "#888888") for t in sorted_tasks
    ]

    ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        alpha=0.85,
        height=0.7,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_tasks, fontsize=8.5)
    ax.set_xlabel("Mean Score ± Std (%)", fontsize=11)
    ax.set_title("MVEB Task Difficulty", fontsize=12)
    ax.set_xlim(0, 108)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    present_types = set(task_type_map[t] for t in sorted_tasks)
    legend_elements = [
        mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_ABBREV[t])
        for t in TYPE_ABBREV
        if t in present_types
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right", frameon=True)

    fig.tight_layout()

    out_pdf = OUT_DIR / "plot_task_difficulty.pdf"
    out_png = OUT_DIR / "plot_task_difficulty.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_pdf}")

    print("\nTask means (ascending — hardest first):")
    for t in sorted_tasks:
        n = task_n.get(t, 0)
        print(f"  {task_means[t]:5.1f}% ± {task_stds.get(t, 0):4.1f}  (n={n:2d})  {t}")


if __name__ == "__main__":
    main()
