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
    type_to_scores: dict[str, list[float]] = defaultdict(list)
    for tr in task_results:
        score = tr.get_score()
        if score is None or np.isnan(score):
            return None
        type_to_scores[tr.task.metadata.type].append(score)
    mean_per_type = {t: sum(s) / len(s) for t, s in type_to_scores.items()}
    return sum(mean_per_type.values()) / len(mean_per_type) if mean_per_type else 0.0
