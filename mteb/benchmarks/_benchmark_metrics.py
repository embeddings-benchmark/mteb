from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mteb.results.task_result import TaskResult


def _compute_mean_task(task_results: list[TaskResult]) -> float | None:
    """Mean score across task results.

    Returns ``None`` if any score is missing or NaN, ``0.0`` if the list is empty.
    """
    all_scores = [tr.get_score() for tr in task_results]
    if any(s is None or np.isnan(s) for s in all_scores):
        return None
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


def _compute_mean_task_type(task_results: list[TaskResult]) -> float | None:
    """Mean of per-task-type means.

    Returns ``None`` if any score is missing or NaN, ``0.0`` if the list is empty.
    """
    mean_per_type = _compute_task_types(task_results)
    if mean_per_type is None:
        return None
    return sum(mean_per_type.values()) / len(mean_per_type) if mean_per_type else 0.0


def _compute_task_types(
    task_results: list[TaskResult],
) -> dict[str, float] | None:
    """Per-task-type mean scores keyed by raw task-type name.

    Returns ``None`` if any score is missing or NaN, ``{}`` if the list is empty.
    """
    type_to_scores: dict[str, list[float]] = defaultdict(list)
    for tr in task_results:
        score = tr.get_score()
        if score is None or np.isnan(score):
            return None
        type_to_scores[tr.task.metadata.type].append(score)
    return {t: sum(s) / len(s) for t, s in type_to_scores.items()}


def _compute_mean_public_private(
    task_results: list[TaskResult],
) -> dict[str, float | None]:
    """Mean score split into the public and private task partitions.

    Each value is ``None`` if any score in that partition is missing/NaN,
    and absent partitions get a ``None`` value so callers see both keys.
    """
    public_scores: list[float] = []
    private_scores: list[float] = []
    public_has_null = False
    private_has_null = False
    for tr in task_results:
        score = tr.get_score()
        is_public = tr.task.metadata.is_public
        bucket = public_scores if is_public else private_scores
        if score is None or np.isnan(score):
            if is_public:
                public_has_null = True
            else:
                private_has_null = True
            continue
        bucket.append(score)

    def _mean(scores: list[float], has_null: bool) -> float | None:
        if has_null:
            return None
        if not scores:
            return None
        return sum(scores) / len(scores)

    return {
        "Mean(Public)": _mean(public_scores, public_has_null),
        "Mean(Private)": _mean(private_scores, private_has_null),
    }


def _compute_mean_subset(
    task_results: list[TaskResult],
) -> dict[str, float | None]:
    """Mean weighted equally across all ``(task, subset)`` entries.

    Mirrors the polars subset-weighted path used by HUME — for each
    ``(task, subset)`` pair, average the per-split main scores; then take
    the unweighted mean across all such pairs.

    Returns:
        dict: `{"Mean(Subset)": value}` where `value` is `None` if any
            subset score is missing/NaN, `0.0` for an empty input list, and
            the subset-weighted mean otherwise.
    """
    by_subset: dict[tuple[str, str], list[float]] = defaultdict(list)
    for tr in task_results:
        for split_scores in tr.scores.values():
            for subset_score in split_scores:
                main = subset_score.get("main_score")
                if main is None or (isinstance(main, float) and np.isnan(main)):
                    return {"Mean(Subset)": None}
                subset_key = subset_score.get("hf_subset", "default")
                by_subset[(tr.task_name, subset_key)].append(float(main))
    if not by_subset:
        return {"Mean(Subset)": 0.0}
    means_per_subset = [sum(s) / len(s) for s in by_subset.values()]
    return {"Mean(Subset)": sum(means_per_subset) / len(means_per_subset)}


def _task_types_or_nulls(
    task_results: list[TaskResult],
) -> dict[str, float | None]:
    """Wrap [_compute_task_types][mteb.benchmarks._benchmark_metrics._compute_task_types] so a missing/NaN score nulls every per-type column.

    Mirrors the all-or-nothing semantics of the scalar aggregators: if any
    task is missing a score, every type column comes back `None` rather
    than `_compute_task_types` returning `None` (which would drop the
    keys entirely).
    """
    result = _compute_task_types(task_results)
    if result is not None:
        return dict(result)
    return {tr.task.metadata.type: None for tr in task_results}
