"""Estimate how much of the score spread between top models on each task is sampling noise.

For every task the standard error (SE) of the main score is estimated from two sources:
- Test-set sampling, from the number of evaluated items (`descriptive_stats`) and the score level:
    - scores bounded in [0, 1] (classification, bitext mining, retrieval, reranking):
      `sqrt(m * (1 - m) / n)`, where `n` is the number of samples or queries. For retrieval
      and reranking this is an upper bound, since per-query scores vary less than a 0/1 variable.
    - STS (Spearman): `(1 - m^2) * sqrt(1.06 / (n - 3))`.
    - Pair classification (AP): `sqrt(m * (1 - m) / n_positive)`.
    - Instruction reranking (p-MRR in [-1, 1]): `1 / sqrt(n_queries)` (upper bound).
- Evaluation randomness, from the result files of the top models:
    - classification: std of the main score over the undersampling experiments / sqrt(n_experiments).
    - clustering: `v_measure_std / sqrt(n_sets)`.

Scores averaged over K subsets/splits get `SE = sqrt(sum SE_i^2) / K` if subsets are independent. Parallel
datasets reuse the same sentences in every subset, so `*_conservative` columns instead assume fully
correlated subsets: `SE = mean(SE_i)`.
Models are evaluated on the same items, so their errors are correlated. The SE of a difference between two
models is approximated as `sqrt(2 * (1 - r)) * SE` with `--paired-correlation r` (default 0.5, giving
`SE_diff = SE`).

Outputs a CSV with, per task:
- `se_sampling`, `se_evaluation`: the two SE sources.
- `se_diff`: estimated SE of a difference between two models.
- `std_top`: std of scores among the top models.
- `signal_to_noise`: `std_top / se_diff`.
- `share_distinguishable`: share of top-model pairs whose score difference exceeds `1.96 * se_diff`.

Example:
    uv run --no-sync python scripts/benchmark_task_noise.py --results-path ~/Desktop/results
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from benchmark_score_spread import load_scores, select_models

import mteb


def _unit_stats(task: mteb.AbsTask) -> list[dict[str, Any]]:
    """Descriptive statistics of every (split, subset) the main score is averaged over."""
    stats = json.loads(task.metadata.descriptive_stat_path.read_text())
    units = []
    for split in task.metadata.eval_splits:
        split_stats = stats[split]
        subsets = split_stats.get("hf_subset_descriptive_stats")
        units.extend(subsets.values() if subsets else [split_stats])
    return units


def _num_items(unit: dict[str, Any], task_type: str) -> int:
    if task_type in ("Retrieval", "Reranking", "InstructionReranking"):
        if "num_queries" in unit:
            return unit["num_queries"]
        # older descriptive statistics don't store the number of queries
        return unit["queries_text_statistics"]["unique_texts"]
    if task_type == "PairClassification":
        labels = unit["labels_statistics"]["labels"]
        return labels.get("1", {}).get("count", unit["num_samples"])
    return unit["num_samples"]


def sampling_se(
    task: mteb.AbsTask, score: float
) -> tuple[float | None, float | None, int | None]:
    """SE of the main score from test-set sampling and the total number of items.

    Returns the SE assuming independent subsets and the SE assuming fully correlated subsets
    (e.g. parallel data where every language pair reuses the same sentences).
    """
    task_type = task.metadata.type
    if task_type == "Clustering":
        return None, None, None
    units = _unit_stats(task)
    variances = []
    total = 0
    for unit in units:
        n = _num_items(unit, task_type)
        total += n
        if task_type == "STS":
            se = (1 - min(score, 0.999) ** 2) * math.sqrt(1.06 / max(n - 3, 1))
        elif task_type == "InstructionReranking":
            se = 1 / math.sqrt(n)
        else:
            m = min(max(score, 0.0), 1.0)
            se = math.sqrt(max(m * (1 - m), 1e-4) / n)
        variances.append(se**2)
    independent = math.sqrt(sum(variances)) / len(variances)
    correlated = float(np.mean(np.sqrt(variances)))
    return independent, correlated, total


def _result_file(results_path: Path, model: str, task_name: str) -> Path | None:
    model_dir = results_path / "results" / model.replace("/", "__")
    files = sorted(model_dir.glob(f"*/{task_name}.json"))
    return files[0] if files else None


def evaluation_se(
    task: mteb.AbsTask, models: list[str], results_path: Path
) -> float | None:
    """Median over models of the SE caused by evaluation randomness (classification and clustering)."""
    task_type = task.metadata.type
    if task_type not in (
        "Classification",
        "MultilabelClassification",
        "Clustering",
    ):
        return None
    main_score = task.metadata.main_score
    per_model = []
    for model in models:
        path = _result_file(results_path, model, task.metadata.name)
        if path is None:
            continue
        scores = json.loads(path.read_text())["scores"]
        variances = []
        for split in task.metadata.eval_splits:
            for subset in scores.get(split, []):
                if task_type == "Clustering":
                    # older results store a list, newer ones a dict of hierarchy level -> list
                    sets = subset.get("v_measures", [])
                    if isinstance(sets, dict):
                        sets = next(iter(sets.values()), [])
                    std = subset.get("v_measure_std")
                    if std is None or len(sets) < 2:
                        continue
                    variances.append(std**2 / len(sets))
                else:
                    experiments = [
                        exp[main_score]
                        for exp in subset.get("scores_per_experiment", [])
                        if main_score in exp
                    ]
                    if len(experiments) < 2:
                        continue
                    variances.append(np.var(experiments, ddof=1) / len(experiments))
        if variances:
            per_model.append(math.sqrt(sum(variances)) / len(variances))
    return float(np.median(per_model)) if per_model else None


def share_distinguishable(scores: pd.Series, se_diff: float) -> float:
    values = scores.dropna().to_numpy()
    pairs = list(itertools.combinations(values, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([abs(a - b) > 1.96 * se_diff for a, b in pairs]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="MTEB(Multilingual, v2)")
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to a clone of embeddings-benchmark/results",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_task_noise.csv"))
    parser.add_argument("--min-coverage", type=float, default=0.97)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--paired-correlation", type=float, default=0.5)
    args = parser.parse_args()

    results_path = args.results_path.expanduser().resolve()
    scores = load_scores(args.benchmark, results_path)
    models = select_models(scores, args.min_coverage, args.top_k)
    paired_factor = math.sqrt(2 * (1 - args.paired_correlation))

    rows = []
    for task in mteb.get_benchmark(args.benchmark).tasks:
        name = task.metadata.name
        top_scores = scores.loc[name, models]
        level = float(top_scores.median())
        se_independent, se_correlated, n_items = sampling_se(task, level)
        se_eval = evaluation_se(task, models, results_path)
        std_top = float(top_scores.std())
        row = {
            "task": name,
            "type": task.metadata.type,
            "main_score": task.metadata.main_score,
            "n_items": n_items,
            "n_subsets": len(_unit_stats(task)),
            "median_top": level,
            "std_top": std_top,
            "se_sampling": se_independent,
            "se_sampling_correlated": se_correlated,
            "se_evaluation": se_eval,
        }
        for suffix, se_sampling in (
            ("", se_independent),
            ("_conservative", se_correlated),
        ):
            if se_sampling is None and se_eval is None:
                se_diff = float("nan")
            else:
                se = math.sqrt((se_sampling or 0) ** 2 + (se_eval or 0) ** 2)
                se_diff = paired_factor * se
            row[f"se_diff{suffix}"] = se_diff
            row[f"signal_to_noise{suffix}"] = std_top / se_diff
            row[f"share_distinguishable{suffix}"] = (
                share_distinguishable(top_scores, se_diff)
                if not math.isnan(se_diff)
                else float("nan")
            )
        rows.append(row)

    result = pd.DataFrame(rows).set_index("task")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(args.output)
    print(f"Using {len(models)} models, wrote {len(result)} tasks to {args.output}")


if __name__ == "__main__":
    main()
