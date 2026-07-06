"""Tests for AbsTaskAggregate"""

import logging

import mteb
from mteb.abstasks.aggregated_task import AbsTaskAggregate

logging.basicConfig(level=logging.INFO)


def test_is_aggregate_property_correct():
    tasks = mteb.get_tasks()

    for task in tasks:
        assert task.is_aggregate == isinstance(task, AbsTaskAggregate)


def test_task_results_to_scores_isolates_splits():
    from mteb.results.task_result import TaskResult
    from tests.mock_tasks import MockAggregatedTask

    agg_task = MockAggregatedTask()
    # To test split isolation, the task MUST have at least two splits.
    # MockAggregatedTask only has 'test' by default, so we inject 'dev' for this test
    # using model_copy() so we don't pollute the global mock class metadata for other tests.
    agg_task.metadata = agg_task.metadata.model_copy(
        update={"eval_splits": ("test", "dev")}
    )

    # The aggregate expects results for its inner tasks. We only need one to prove the math.
    inner_task_name = agg_task.tasks[0].metadata.name

    mock_task_result = TaskResult(
        dataset_revision="1.0",
        task_name=inner_task_name,
        mteb_version="1.0",
        scores={
            "test": [
                {
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                    "main_score": 0.8,
                    agg_task.metadata.main_score: 0.8,
                    "mteb_version": "1.0",
                }
            ],
            "dev": [
                {
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                    "main_score": 0.5,
                    agg_task.metadata.main_score: 0.5,
                    "mteb_version": "1.0",
                }
            ],
        },
        evaluation_time=1.0,
    )

    scores = agg_task.task_results_to_scores([mock_task_result])

    assert scores["test"]["default"]["main_score"] == 0.8
    assert scores["dev"]["default"]["main_score"] == 0.5
