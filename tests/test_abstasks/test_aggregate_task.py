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
    from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.abstasks.abstask import AbsTask
    from mteb.results.task_result import TaskResult

    mock_task_meta = TaskMetadata(
        name="MockTask1",
        description="Mock",
        dataset={"path": "mock", "revision": "1.0"},
        reference=None,
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "dev"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    class MockTask(AbsTask):
        metadata = mock_task_meta
        def _evaluate_subset(self, *args, **kwargs): pass
        def _calculate_descriptive_statistics_from_split(self, *args, **kwargs): pass

    mock_task = MockTask()

    mock_task_result = TaskResult(
        dataset_revision="1.0",
        task_name="MockTask1",
        mteb_version="1.0",
        scores={
            "test": [
                {"hf_subset": "default", "languages": ["eng-Latn"], "main_score": 0.8, "accuracy": 0.8, "mteb_version": "1.0"}
            ],
            "dev": [
                {"hf_subset": "default", "languages": ["eng-Latn"], "main_score": 0.5, "accuracy": 0.5, "mteb_version": "1.0"}
            ]
        },
        evaluation_time=1.0,
    )

    agg_meta = AggregateTaskMetadata(
        name="AggTask",
        description="Mock agg",
        reference=None,
        dataset={"path": "mock", "revision": "1.0"},
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "dev"],
        eval_langs={"default": ["eng-Latn"]},
        main_score="accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        tasks=[mock_task]
    )

    class MockAggTask(AbsTaskAggregate):
        metadata = agg_meta

    agg_task = MockAggTask()
    scores = agg_task.task_results_to_scores([mock_task_result])

    assert scores["test"]["default"]["main_score"] == 0.8
    assert scores["dev"]["default"]["main_score"] == 0.5
