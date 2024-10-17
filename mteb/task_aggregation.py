from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from mteb.load_results.benchmark_results import BenchmarkResults
from mteb.load_results.task_results import TaskResult
from mteb.overview import get_task

logger = logging.getLogger(__name__)

REVISION = str
MODEL_NAME = str
AGGREGATION = dict[MODEL_NAME, dict[REVISION, dict[str, float]]]


def mean(results: BenchmarkResults) -> AGGREGATION:
    """Calculate the mean of the main score of the given results."""
    results = results.to_legacy_dict()
    unique_tasks = set()
    for model, revisions in results.items():
        for revision, res in revisions.items():
            for result in res:
                unique_tasks.add(result.task_name)

    def _mean(model_name: str, rev: str, results: list[TaskResult]) -> float:
        """Calculate the mean of the main score of the given results."""
        scores: list[float] = [result.get_score() for result in results]

        if len(scores) != len(unique_tasks):
            logger.warning(
                f"Model {model_name} revision {rev} has missing scores for some tasks"
            )

        if scores:
            return sum(scores) / len(unique_tasks)
        return np.nan

    return {
        model: {rev: {"mean": _mean(model, rev, res)} for rev, res in revs.items()}
        for model, revs in results.items()
    }


def task_category_weighted_mean(
    results: BenchmarkResults,
) -> AGGREGATION:
    """Calculate the mean of the main score of the given results, weighted by the number of tasks of each type."""
    results = results.to_legacy_dict()
    unique_tasks = set()
    task_types = defaultdict(set)
    for model, revisions in results.items():
        for revision, res in revisions.items():
            for result in res:
                task_name = result.task_name
                task_type = get_task(task_name).metadata.type
                unique_tasks.add(task_name)
                task_types[task_type].add(task_name)

    def _task_category_weighted_mean(
        model: str, rev: str, results: list[TaskResult]
    ) -> dict[str, float]:
        """Calculate the mean of the main score of the given results, weighted by the number of tasks of each type."""
        _task_types = {task_type: [] for task_type in task_types.keys()}

        for result in results:
            task_name = result.task_name
            task_type = get_task(task_name).metadata.type
            _task_types[task_type].append(result.get_score())

        # mean pr task type then mean of means
        means = {}
        for task_type, scores in _task_types.items():
            if len(scores) != len(task_types[task_type]):
                logger.warning(
                    f"Model {model} revision {rev} has missing scores for some tasks of type {task_type}"
                )
            _mean = sum(scores) / len(task_types[task_type]) if scores else np.nan
            # means.append(_mean)
            means[f"mean ({task_type})"] = _mean

        _mean = sum(means.values()) / len(task_types)
        means["mean (weighted by task type)"] = _mean
        return means

    return {
        model: {
            rev: _task_category_weighted_mean(model, rev, res)
            for rev, res in revs.items()
        }
        for model, revs in results.items()
    }


def borda_count(
    results: BenchmarkResults,
) -> AGGREGATION:
    """Calculate the Borda count of the given results.

    To handle ties, we use the [Tournament Borda Count method](https://en.wikipedia.org/wiki/Borda_count#Equal_ranks).
    This method assigns the average of the ranks that would have been assigned to the tied candidates to each of the tied candidates. So if two
    candidates would otherwise have gained 1 or 2 points (if not tied), they both gain 1.5 points.
    """
    # consider each model a candidate and each task a voter
    # each voter ranks the candidates

    results = results.to_legacy_dict()
    n_candidates = sum(len(revs) for revs in results.values())
    candidate_scores = {
        model: {revision: 0.0 for revision in revisions}
        for model, revisions in results.items()
    }

    tasks = defaultdict(list)  # {task_name: [(model, revision, score), ...]}

    for model, revisions in results.items():
        for revision, task_results in revisions.items():
            for task_result in task_results:
                task_name = task_result.task_name
                score = task_result.get_score()
                tasks[task_name].append((model, revision, score))

    for task_name, task_results in tasks.items():
        task_results.sort(key=lambda x: x[2])
        # scores to assign to each candidate
        scores = list(range(0, n_candidates, 1))

        prev_result = None
        tied_group = []
        score = 0
        while task_results:
            _result = task_results.pop()

            if (prev_result is None) or (_result[2] == prev_result[2]):
                score += scores.pop()
                tied_group.append(_result)
            else:  # resolve point assignment
                for task_result in tied_group:
                    candidate_scores[task_result[0]][task_result[1]] += score / len(
                        tied_group
                    )
                tied_group = [_result]
                score = scores.pop()
            prev_result = _result

        # resolve last group
        for task_result in tied_group:
            candidate_scores[task_result[0]][task_result[1]] += score / len(tied_group)

    return {
        model: {rev: {"borda_count": score} for rev, score in revs.items()}
        for model, revs in candidate_scores.items()
    }


aggregation_methods = {
    "Mean (naïve)": mean,
    "Mean (weighted by task category)": task_category_weighted_mean,
    "Rank (Borda Count)": borda_count,
}
