#!/usr/bin/env python3
"""Analyze saved MovingFashion V2I pair-classification predictions on CPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.metrics import average_precision_score

from mteb._evaluators.pair_classification_evaluator import (
    PairClassificationDistances,
)
from mteb.tasks.pair_classification.zxx.moving_fashion_pc import (
    MovingFashionV2IPairClassification,
)

_SCORE_DIRECTIONS = {
    "similarity_scores": True,
    "cosine_scores": True,
    "manhattan_distances": False,
    "euclidean_distances": False,
    "dot_scores": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-trials", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.random_trials < 1:
        parser.error("--random-trials must be positive")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


def _find_score_block(value: Any) -> PairClassificationDistances:
    if isinstance(value, dict):
        if _SCORE_DIRECTIONS.keys() <= value.keys():
            return cast(
                "PairClassificationDistances",
                {name: value[name] for name in _SCORE_DIRECTIONS},
            )
        for child in value.values():
            try:
                return _find_score_block(child)
            except ValueError:
                pass
    elif isinstance(value, list):
        for child in value:
            try:
                return _find_score_block(child)
            except ValueError:
                pass
    raise ValueError("Could not find pair-classification scores in prediction file")


def _slice_scores(
    scores: PairClassificationDistances, indices: np.ndarray
) -> PairClassificationDistances:
    return cast(
        "PairClassificationDistances",
        {name: np.asarray(values)[indices].tolist() for name, values in scores.items()},
    )


def _max_ap(scores: PairClassificationDistances, labels: np.ndarray) -> float:
    score_values = cast("dict[str, list[float]]", scores)
    return float(
        max(
            average_precision_score(
                labels,
                np.asarray(score_values[name]) * (1 if higher_is_better else -1),
            )
            for name, higher_is_better in _SCORE_DIRECTIONS.items()
        )
    )


def _summary(values: list[float], *, interval_name: str) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        interval_name: [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def _random_ranking_baseline(
    task: MovingFashionV2IPairClassification,
    labels: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(trials):
        random_scores = rng.random(len(labels)).tolist()
        metrics = task._compute_metrics_values(random_scores, labels, True)
        for name, value in metrics.items():
            values[name].append(value)
    return {
        "trials": trials,
        "seed": seed,
        "description": "Independent random pair ranking; one score per pair.",
        "metrics": {
            name: _summary(metric_values, interval_name="central_95_interval")
            for name, metric_values in values.items()
        },
    }


def _cluster_bootstrap(
    scores: PairClassificationDistances,
    labels: np.ndarray,
    video_ids: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    row_indices_by_video: dict[str, list[int]] = defaultdict(list)
    for row_index, video_id in enumerate(video_ids):
        row_indices_by_video[video_id].append(row_index)

    unique_video_ids = list(row_indices_by_video)
    rng = np.random.default_rng(seed)
    bootstrap_values = []
    for _ in range(samples):
        sampled_video_indices = rng.integers(
            0, len(unique_video_ids), size=len(unique_video_ids)
        )
        row_indices = np.concatenate(
            [
                row_indices_by_video[unique_video_ids[index]]
                for index in sampled_video_indices
            ]
        )
        bootstrap_values.append(
            _max_ap(_slice_scores(scores, row_indices), labels[row_indices])
        )

    return {
        "samples": samples,
        "seed": seed,
        "cluster": "video_id",
        "metric": "max_ap",
        "point_estimate": _max_ap(scores, labels),
        **_summary(bootstrap_values, interval_name="percentile_95_confidence_interval"),
    }


def analyze_predictions(
    predictions_path: Path,
    pairs_path: Path,
    *,
    random_trials: int = 1_000,
    bootstrap_samples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text())
    pairs = json.loads(pairs_path.read_text())
    scores = _find_score_block(predictions)
    score_values = cast("dict[str, list[float]]", scores)
    rows = pairs["rows"]
    labels = np.asarray(rows["label"], dtype=np.int64)
    video_ids = rows["video_id"]
    sources = rows["source_subset"]

    lengths = {
        len(labels),
        len(video_ids),
        len(rows["image_id"]),
        len(sources),
        *(len(values) for values in score_values.values()),
    }
    if len(lengths) != 1:
        raise ValueError(f"Prediction and pair-manifest lengths differ: {lengths}")
    if set(labels) != {0, 1}:
        raise ValueError("Pair manifest must contain both binary labels 0 and 1")

    task = MovingFashionV2IPairClassification()
    overall = task._compute_metrics(scores, labels.tolist())
    by_source = {}
    for source in sorted(set(sources)):
        indices = np.flatnonzero(np.asarray(sources) == source)
        source_labels = labels[indices]
        by_source[source] = {
            "num_pairs": len(indices),
            "label_counts": {
                str(label): count
                for label, count in sorted(Counter(source_labels).items())
            },
            "metrics": task._compute_metrics(
                _slice_scores(scores, indices), source_labels.tolist()
            ),
        }

    return {
        "task_name": pairs["task_name"],
        "dataset_revision": pairs["dataset_revision"],
        "num_pairs": len(labels),
        "num_unique_videos": len(set(video_ids)),
        "num_unique_images": len(set(rows["image_id"])),
        "label_counts": {
            str(label): count for label, count in sorted(Counter(labels).items())
        },
        "overall": overall,
        "by_source": by_source,
        "random_ranking": _random_ranking_baseline(
            task, labels, trials=random_trials, seed=seed
        ),
        "video_cluster_bootstrap": _cluster_bootstrap(
            scores,
            labels,
            video_ids,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def write_analysis(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")


def print_summary(analysis: dict[str, Any]) -> None:
    print(f"overall_max_ap={analysis['overall']['max_ap']:.6f}")
    for source, result in analysis["by_source"].items():
        print(f"{source}_max_ap={result['metrics']['max_ap']:.6f}")
    random_ap = analysis["random_ranking"]["metrics"]["ap"]
    print(f"random_ap_mean={random_ap['mean']:.6f}")
    print(f"random_ap_central_95_interval={random_ap['central_95_interval']}")
    bootstrap = analysis["video_cluster_bootstrap"]
    print(
        f"max_ap_video_bootstrap_95_ci={bootstrap['percentile_95_confidence_interval']}"
    )


def main() -> None:
    args = _parse_args()
    output_path = args.output or args.predictions.with_name(
        "MovingFashionV2IPairClassification_analysis.json"
    )
    analysis = analyze_predictions(
        args.predictions,
        args.pairs,
        random_trials=args.random_trials,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_analysis(analysis, output_path)
    print_summary(analysis)
    print(f"analysis_file={output_path.resolve()}")


if __name__ == "__main__":
    main()
