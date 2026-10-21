"""Find benchmark tasks where most models have very similar scores.

For every task, scores of all models covering the benchmark are compared to their median:
- `within`: share of models scoring within `--tolerance` of the median.
- `breakout_models`: models scoring more than `--breakout` above the median.

`concentrated` marks tasks with `within >= --min-within`. Tasks are sorted by `within`.

Outputs a CSV and a markdown table with all tasks.

Example:
    uv run --no-sync python scripts/benchmark_score_concentration.py --results-path ~/Desktop/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from benchmark_score_spread import load_scores, select_models


def compute_concentration(
    scores: pd.DataFrame,
    models: list[str],
    tolerance: float,
    breakout: float,
    min_within: float,
) -> pd.DataFrame:
    rows = []
    for task, task_scores in scores[models].iterrows():
        values = task_scores.dropna()
        median = values.median()
        within = ((values - median).abs() <= tolerance).mean()
        breakouts = values[values - median > breakout].sort_values(ascending=False)
        rows.append(
            {
                "task": task,
                "concentrated": within >= min_within,
                "within": within,
                "median": median,
                "p10": values.quantile(0.1),
                "p90": values.quantile(0.9),
                "max": values.max(),
                "n_models": len(values),
                "n_breakout": len(breakouts),
                "breakout_models": ", ".join(
                    f"{model.split('/')[-1]} {score:.3f}"
                    for model, score in breakouts.items()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("task").sort_values("within", ascending=False)


def to_markdown(table: pd.DataFrame) -> str:
    table = table.reset_index()
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for row in table.itertuples(index=False):
        cells = [
            f"{value:.3f}" if isinstance(value, float) else str(value) for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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
        "--output", type=Path, default=Path("benchmark_score_concentration.csv")
    )
    parser.add_argument("--min-coverage", type=float, default=0.97)
    parser.add_argument(
        "--top-k", type=int, default=0, help="Number of top models, 0 for all"
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--breakout", type=float, default=0.1)
    parser.add_argument("--min-within", type=float, default=0.5)
    args = parser.parse_args()

    scores = load_scores(args.benchmark, args.results_path)
    models = select_models(scores, args.min_coverage, args.top_k)
    table = compute_concentration(
        scores, models, args.tolerance, args.breakout, args.min_within
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.output)
    md_output = args.output.with_suffix(".md")
    md_output.write_text(
        f"# {args.benchmark}: score concentration over {len(models)} models\n\n"
        f"`within`: share of models within {args.tolerance} of the median. "
        f"`breakout_models`: models more than {args.breakout} above the median.\n\n"
        + to_markdown(table)
    )
    print(
        f"Using {len(models)} models, wrote {len(table)} tasks to {args.output} and {md_output}"
    )


if __name__ == "__main__":
    main()
