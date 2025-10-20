"""Tests the TaskMetadata class"""

import pytest
from pydantic import ValidationError

from mteb.abstasks.task_metadata import TaskMetadata
from tests.task_grid import ALL_TASK_TEST_GRID


@pytest.mark.parametrize("task", ALL_TASK_TEST_GRID)
def test_descriptive_stats(task):
    result_stat = task.calculate_descriptive_statistics()
    # remove descriptive task file
    task.metadata.descriptive_stat_path.unlink()
    task_stat = task.expected_stats
    print(task.metadata.name)
    print(result_stat)

    for key, value in result_stat.items():
        assert key in task_stat
        assert value == task_stat[key]


def test_given_dataset_config_then_it_is_valid():
    my_task = TaskMetadata(
        name="MyTask",
        dataset={
            "path": "test/dataset",
            "revision": "1.0",
        },
        description="testing",
        reference=None,
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=None,
        domains=None,
        license=None,
        task_subtypes=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
    )
    assert my_task.dataset["path"] == "test/dataset"
    assert my_task.dataset["revision"] == "1.0"


def test_given_missing_dataset_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(  # type: ignore
            name="MyTask",
            description="testing",
            reference=None,
            type="Classification",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map_at_1000",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_missing_revision_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map_at_1000",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_none_revision_path_then_it_logs_warning(caplog):
    with pytest.raises(ValidationError):
        TaskMetadata(
            name="MyTask",
            dataset={"path": "test/dataset", "revision": None},
            description="testing",
            reference=None,
            type="Classification",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map_at_1000",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_unfilled_metadata_is_not_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map_at_1000",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        ).is_filled()
        is False
    )


def test_filled_metadata_is_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference="https://aclanthology.org/W19-6138/",
            type="Classification",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map_at_1000",
            date=("2021-01-01", "2021-12-31"),
            domains=["Non-fiction", "Written"],
            license="mit",
            task_subtypes=["Thematic clustering"],
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
            bibtex_citation="Someone et al",
        ).is_filled()
        is True
    )
