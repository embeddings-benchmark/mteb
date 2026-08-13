from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.benchmarks.benchmark import CustomGrouping
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


def _bucket_task_result_scores(
    task_results: list[TaskResult],
    key_fn: Callable[[TaskResult], str | None],
) -> tuple[dict[str, list[float]], bool]:
    """Bucket ``task_results``' scores by ``key_fn(tr)``.

    A result whose ``key_fn`` returns ``None`` is skipped entirely — not
    bucketed and not counted toward the null flag. Otherwise a missing/NaN
    score sets ``saw_null`` but the result is still excluded from its bucket.

    Shared primitive behind [_compute_task_types][mteb.benchmarks._benchmark_metrics._compute_task_types]
    (keyed by ``task.metadata.type``, every result is bucketed) and
    [_compute_custom_group_means][mteb.benchmarks._benchmark_metrics._compute_custom_group_means]
    (keyed by ``grouping.task_to_label``, results outside the dimension are
    skipped) — the `dict`-based analog of
    [_bucket_means][mteb.benchmarks._benchmark_metrics._bucket_means] for
    callers that still hold `TaskResult` objects rather than a flat
    `{task_name: score}` dict.

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


def _compute_custom_group_means(
    task_results: list[TaskResult], grouping: CustomGrouping
) -> dict[str, float | None]:
    """Per-custom-group mean scores, namespaced ``"{dimension}::{label}"``.

    Mirrors [_task_types_or_nulls][mteb.benchmarks._benchmark_metrics._task_types_or_nulls]'s
    all-or-nothing semantics, scoped to the tasks this dimension actually
    covers: if any task assigned to a group in `grouping` is missing a
    score, every group in this dimension comes back `None`. Tasks not
    assigned to any group in `grouping` (a dimension may cover only a
    subset of the benchmark's tasks) are ignored entirely. A group with no
    scored tasks (e.g. its declared tasks aren't part of this benchmark)
    gets `0.0`, matching [_compute_mean_subset][mteb.benchmarks._benchmark_metrics._compute_mean_subset]'s
    empty-input fallback.
    """
    buckets, has_null = _bucket_task_result_scores(
        task_results, lambda tr: grouping.task_to_label.get(tr.task.metadata.name)
    )

    result: dict[str, float | None] = {}
    for group in grouping.groups:
        key = f"{grouping.name}::{group.label}"
        if has_null:
            result[key] = None
            continue
        scores = buckets.get(group.label, [])
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
    """Average `scores_by_task` grouped by `task_to_key`.

    Keeps only tasks present in `scores_by_task` — a task missing a score
    (filtered out / not run) is absent from its bucket rather than treated
    as a zero, and a task with no key (`task_to_key.get` misses) is skipped
    entirely.

    Shared bucket-and-average primitive behind both
    [_recompute_lenient_means][mteb.benchmarks._benchmark_metrics._recompute_lenient_means]
    (bucketed by task type) and
    [_recompute_lenient_custom_groups][mteb.benchmarks._benchmark_metrics._recompute_lenient_custom_groups]
    (bucketed by CustomGrouping label, once per dimension) — same
    "only average what's present" policy either way, so partial-coverage
    models under a language filter don't collapse to null.
    """
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
    """Recompute means over only the tasks a model actually ran.

    Why: used by the API's language-filtered path (`mteb.api.aggregators`)
    so partial-coverage models don't collapse to null. Per-task-type
    bucketing (via `_bucket_means`) prevents a single dense type (e.g.
    Classification with 20 tasks) from outweighing sparser ones.
    """
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

    The CustomGrouping analog of `_recompute_lenient_means` — one
    `_bucket_means` call per dimension instead of a single task-type bucketing.
    Used by the API's language-filtered path (`mteb.api.aggregators`).

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
