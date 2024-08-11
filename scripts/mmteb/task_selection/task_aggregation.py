from __future__ import annotations

from collections import defaultdict

from mteb.load_results.mteb_results import MTEBResults
from mteb.overview import get_task


def mean(results: list[MTEBResults]) -> float:
    """Calculate the mean of the main score of the given results."""
    scores: list[float] = [result.get_score() for result in results]
    return sum(scores) / len(scores)


def task_category_weighted_mean(results: list[MTEBResults]) -> float:
    """Calculate the mean of the main score of the given results, weighted by the number of tasks of each type."""
    task_types = defaultdict(list)

    for result in results:
        task_name = result.task_name
        task_type = get_task(task_name).metadata.type
        task_types[task_type].append(result.get_score())

    # mean pr task type then mean of means
    means = [sum(scores) / len(scores) for scores in task_types.values()]
    return sum(means) / len(means)


def borda_count(results: list[MTEBResults]) -> float:
    """Calculate the Borda count of the given results."""
    scores = [result.get_score() for result in results]
    scores.sort(reverse=True)
    return sum(i * score for i, score in enumerate(scores, start=1)) / len(scores)


# aggregation_methods = {
#     "Mean (naïve)": mean,
#     "Mean (weighted by task category)": task_category_weighted_mean,
#     "Rank (Borda Count)": borda_count,
#     # "Generalization factor": None,
# }
