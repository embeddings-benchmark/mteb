"""Find benchmark tasks with low score diversity and tasks dominated by a single outlier model.

For every task, models are sorted by their score on that task and the following are computed:
- `diff_1_50`: score of the best model minus the score of the 50th best model.
- `diff_10_50`: score of the 10th best model minus the score of the 50th best model.
- `jump_1_10`: score of the best model minus the score of the 10th best model.

Outputs one CSV with all tasks, where `low_diversity` marks `diff_1_50 <= --spread-threshold`
and `outlier_jump` marks `jump_1_10 >= --jump-threshold`.

Example:
    uv run --no-sync python scripts/benchmark_score_spread.py --results-path ~/Desktop/results
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

import mteb

RANDOM_MODEL = "mteb/baseline-random-encoder"


def load_scores(benchmark_name: str, results_path: Path) -> pd.DataFrame:
    """Load main scores as a (task x model) dataframe from a clone of the results repository."""
    with tempfile.TemporaryDirectory() as tmp:
        # ResultCache expects the results repository to live under `<cache_path>/remote`
        (Path(tmp) / "remote").symlink_to(results_path.expanduser().resolve())
        cache = mteb.ResultCache(tmp)
        benchmark = mteb.get_benchmark(benchmark_name)
        results = cache.load_results(tasks=benchmark.tasks, only_main_score=True)
        df = results.to_dataframe(aggregation_level="task")
    df = df.set_index("task_name")
    return df.select_dtypes(include="number")


def select_models(scores: pd.DataFrame, min_coverage: float, top_k: int) -> list[str]:
    """Select the top models by mean per-task percentile rank among models covering the benchmark."""
    models = [
        model
        for model in scores.columns
        if scores[model].notna().mean() >= min_coverage and model != RANDOM_MODEL
    ]
    mean_rank = scores[models].rank(axis=1, pct=True).mean()
    ranked = mean_rank.sort_values(ascending=False).index.tolist()
    return ranked[:top_k] if top_k > 0 else ranked


def compute_spread(
    scores: pd.DataFrame, min_coverage: float, top_k: int
) -> pd.DataFrame:
    """Compute per-task score spread among models that cover most of the benchmark."""
    models = [
        model
        for model in scores.columns
        if scores[model].notna().mean() >= min_coverage and model != RANDOM_MODEL
    ]
    rows = []
    for task, task_scores in scores[models].iterrows():
        ranked = task_scores.dropna().sort_values(ascending=False)
        if len(ranked) < top_k:
            continue
        top1, top10, top_k_score = (
            ranked.iloc[0],
            ranked.iloc[9],
            ranked.iloc[top_k - 1],
        )
        rows.append(
            {
                "task": task,
                "top1_model": ranked.index[0],
                "top1": top1,
                "top10": top10,
                f"top{top_k}": top_k_score,
                f"diff_1_{top_k}": top1 - top_k_score,
                f"diff_10_{top_k}": top10 - top_k_score,
                "jump_1_10": top1 - top10,
                f"std_top{top_k}": ranked.iloc[:top_k].std(),
                "random": scores.loc[task, RANDOM_MODEL]
                if RANDOM_MODEL in scores.columns
                else float("nan"),
                "n_models": len(ranked),
            }
        )
    print(f"Using {len(models)} models with coverage >= {min_coverage}")
    return pd.DataFrame(rows).set_index("task")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="MTEB(Multilingual, v2)")
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to a clone of embeddings-benchmark/results",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_score_spread.csv")
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.97,
        help="Minimal share of benchmark tasks a model must have results for",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--spread-threshold", type=float, default=0.05)
    parser.add_argument("--jump-threshold", type=float, default=0.25)
    args = parser.parse_args()

    scores = load_scores(args.benchmark, args.results_path)
    spread = compute_spread(scores, args.min_coverage, args.top_k)

    spread["low_diversity"] = spread[f"diff_1_{args.top_k}"] <= args.spread_threshold
    spread["outlier_jump"] = spread["jump_1_10"] >= args.jump_threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spread.round(4).to_csv(args.output)
    print(f"Wrote {len(spread)} tasks to {args.output}")


if __name__ == "__main__":
    main()
