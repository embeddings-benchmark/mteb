"""Find groups of redundant tasks in a benchmark and plot the task correlation matrix.

Tasks are correlated by how they rank models (Spearman). Only the top `--top-k` models are used, since
including weak models makes almost all tasks correlate. Tasks are grouped with average-linkage
hierarchical clustering on `1 - correlation`, so every group has mean pairwise correlation of at least
`--threshold`.

Outputs:
- `task_correlation.csv`: the task x task correlation matrix.
- `task_groups.csv`: groups of correlated tasks (only groups with more than one task).
- `task_correlation.png`: heatmap with tasks ordered by the clustering.

Example:
    uv run --no-sync --with matplotlib python scripts/benchmark_task_correlation.py --results-path ~/Desktop/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_score_spread import load_scores, select_models
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

import mteb


def cluster_tasks(corr: pd.DataFrame, threshold: float) -> tuple[np.ndarray, list[int]]:
    """Cluster tasks on 1 - correlation and return cluster labels and dendrogram order."""
    distance = (1 - corr.fillna(0)).clip(lower=0).to_numpy()
    np.fill_diagonal(distance, 0)
    links = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(links, t=1 - threshold, criterion="distance")
    return labels, leaves_list(links).tolist()


def build_groups(
    corr: pd.DataFrame, labels: np.ndarray, task_types: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for label in np.unique(labels):
        tasks = corr.index[labels == label].tolist()
        if len(tasks) < 2:
            continue
        sub = corr.loc[tasks, tasks].to_numpy()
        pairwise = sub[np.triu_indices(len(tasks), k=1)]
        rows.append(
            {
                "n_tasks": len(tasks),
                "mean_corr": pairwise.mean(),
                "min_corr": pairwise.min(),
                "task_types": ", ".join(sorted({task_types[t] for t in tasks})),
                "tasks": ", ".join(tasks),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_tasks", "mean_corr"], ascending=False)
        .reset_index(drop=True)
    )


def plot_heatmap(
    corr: pd.DataFrame,
    order: list[int],
    labels: np.ndarray,
    task_types: dict[str, str],
    title: str,
    output: Path,
) -> None:
    ordered = corr.iloc[order, order]
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(0.2 * n + 4, 0.2 * n + 3))
    image = ax.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    tick_labels = [f"{task} ({task_types[task]})" for task in ordered.index]
    ax.set_xticks(range(n), tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n), tick_labels, fontsize=6)

    # outline groups of correlated tasks
    ordered_labels = labels[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or ordered_labels[i] != ordered_labels[start]:
            if i - start > 1:
                ax.add_patch(
                    plt.Rectangle(
                        (start - 0.5, start - 0.5),
                        i - start,
                        i - start,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.2,
                    )
                )
            start = i

    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="Spearman correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="MTEB(Multilingual, v2)")
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to a clone of embeddings-benchmark/results",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--min-coverage", type=float, default=0.97)
    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of top models, 0 for all"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimal mean pairwise correlation inside a group",
    )
    args = parser.parse_args()

    scores = load_scores(args.benchmark, args.results_path)
    models = select_models(scores, args.min_coverage, args.top_k)
    task_types = {
        task.metadata.name: task.metadata.type
        for task in mteb.get_benchmark(args.benchmark).tasks
    }

    corr = scores.loc[:, models].T.corr(method="spearman", min_periods=10)
    labels, order = cluster_tasks(corr, args.threshold)
    groups = build_groups(corr, labels, task_types)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corr.round(4).to_csv(args.output_dir / "task_correlation.csv")
    groups.round(3).to_csv(args.output_dir / "task_groups.csv", index=False)
    plot_heatmap(
        corr,
        order,
        labels,
        task_types,
        f"{args.benchmark}: task correlation over top {len(models)} models",
        args.output_dir / "task_correlation.png",
    )

    print(f"Using {len(models)} models, {len(groups)} groups")
    with pd.option_context(
        "display.width", 300, "display.max_colwidth", None, "display.max_rows", None
    ):
        print(groups.round(3))


if __name__ == "__main__":
    main()
