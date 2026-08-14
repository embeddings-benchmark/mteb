from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.abstasks.abstask import AbsTask
    from mteb.benchmarks.benchmark import CustomGrouping
    from mteb.results.task_result import TaskResult


def _is_whole_task_ref(task_ref: AbsTask) -> bool:
    """True if `task_ref` covers every subset and every split of its task class.

    `hf_subsets`/`eval_splits` default to the full metadata list when
    unfiltered, so a narrower `get_task(hf_subsets=..., eval_splits=...)`
    instance simply won't set-equal them — no `is None` check needed. Lives
    here rather than in `benchmark.py` to avoid an import cycle (that module
    already imports `_compute_custom_group_means` from here).
    """
    return set(task_ref.hf_subsets) == set(task_ref.metadata.hf_subsets) and set(
        task_ref.eval_splits
    ) == set(task_ref.metadata.eval_splits)


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


def _bucket_task_result_scores(
    task_results: list[TaskResult],
    key_fn: Callable[[TaskResult], str | None],
) -> tuple[dict[str, list[float]], bool]:
    """Bucket ``task_results``' scores by ``key_fn(tr)``.

    A result whose ``key_fn`` returns ``None`` is skipped entirely — not
    bucketed and not counted toward the null flag. Otherwise a missing/NaN
    score sets ``saw_null`` but the result is still excluded from its bucket.

    Returns:
        tuple: ``(buckets, saw_null)`` — ``buckets`` maps each observed key
            to its list of (non-null) scores; ``saw_null`` is `True` if any
            *bucketed* result had a missing/NaN score.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    saw_null = False
    for tr in task_results:
        key = key_fn(tr)
        if key is None:
            continue
        score = tr.get_score()
        if score is None or np.isnan(score):
            saw_null = True
            continue
        buckets[key].append(score)
    return buckets, saw_null


def _compute_task_types(
    task_results: list[TaskResult],
) -> dict[str, float] | None:
    """Per-task-type mean scores keyed by raw task-type name.

    Returns ``None`` if any score is missing or NaN, ``{}`` if the list is empty.
    """
    type_to_scores, saw_null = _bucket_task_result_scores(
        task_results, lambda tr: tr.task.metadata.type
    )
    if saw_null:
        return None
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


def _score_for_task_ref(
    by_name: dict[str, TaskResult], task_ref: AbsTask
) -> float | None:
    """One `CustomGroup.tasks` entry -> one scalar score, or `None` if unresolvable.

    Whole-task entries use `tr.get_score()`; scoped entries use
    `tr._get_score_fast(splits=..., subsets=...)`, which averages across
    exactly the given cells so a multi-cell entry still counts as one data
    point. `ValueError` (missing split/subset/score) becomes `None`.
    """
    tr = by_name.get(task_ref.metadata.name)
    if tr is None:
        return None
    try:
        score = (
            tr.get_score()
            if _is_whole_task_ref(task_ref)
            else tr._get_score_fast(
                splits=task_ref.eval_splits, subsets=task_ref.hf_subsets
            )
        )
    except ValueError:
        return None
    return None if (score is None or np.isnan(score)) else score


def _compute_custom_group_means(
    task_results: list[TaskResult], grouping: CustomGrouping
) -> dict[str, float | None]:
    """Per-custom-group mean scores, namespaced ``"{dimension}::{label}"``.

    All-or-nothing: if any `CustomGroup.tasks` entry in `grouping` is
    missing a score, every group in this dimension comes back `None`. A
    group with no scored entries gets `0.0`.
    """
    by_name = {tr.task.metadata.name: tr for tr in task_results}
    has_null = False
    per_group_scores: dict[str, list[float]] = {}
    for group in grouping.groups:
        scores: list[float] = []
        for task_ref in group.tasks:
            score = _score_for_task_ref(by_name, task_ref)
            if score is None:
                has_null = True
            else:
                scores.append(score)
        per_group_scores[group.label] = scores

    result: dict[str, float | None] = {}
    for group in grouping.groups:
        key = f"{grouping.name}::{group.label}"
        if has_null:
            result[key] = None
            continue
        scores = per_group_scores[group.label]
        result[key] = sum(scores) / len(scores) if scores else 0.0
    return result


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


def _bucket_means(
    scores_by_task: dict[str, float], task_to_key: dict[str, str]
) -> dict[str, float]:
    """Average `scores_by_task` grouped by `task_to_key`."""
    buckets: dict[str, list[float]] = {}
    for tname, score in scores_by_task.items():
        key = task_to_key.get(tname)
        if key is None:
            continue
        buckets.setdefault(key, []).append(float(score))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items() if vals}


def _recompute_lenient_means(
    scores_by_task: dict[str, float],
    task_to_type: dict[str, str],
) -> tuple[dict[str, float], float | None, float | None]:
    """Recompute means over only the tasks a model actually ran."""
    scores_by_task_type = _bucket_means(scores_by_task, task_to_type)
    task_vals = list(scores_by_task.values())
    mean_task = sum(task_vals) / len(task_vals) if task_vals else None
    type_vals = list(scores_by_task_type.values())
    mean_type = sum(type_vals) / len(type_vals) if type_vals else None
    return scores_by_task_type, mean_task, mean_type


def _recompute_lenient_custom_groups(
    scores_by_task: dict[str, float],
    custom_group_task_to_label: dict[str, dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Recompute scores_by_custom_group over only the tasks a model actually ran.

    Args:
        scores_by_task: {task_name: score} for tasks the model has a value
            for after the language filter (same input `_recompute_lenient_means`
            takes).
        custom_group_task_to_label: {dimension_name: {task_name: label}} —
            one `CustomGrouping.task_to_label` per dimension the benchmark
            declares.
    """
    return {
        dim: _bucket_means(scores_by_task, task_to_label)
        for dim, task_to_label in custom_group_task_to_label.items()
    }
