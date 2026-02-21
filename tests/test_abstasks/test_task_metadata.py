"""Tests the TaskMetadata class"""

import pytest
from pydantic import ValidationError

import mteb
from mteb.abstasks.task_metadata import TaskMetadata
from tests.task_grid import (
    MOCK_MAEB_TASK_GRID,
    MOCK_MIEB_TASK_GRID,
    MOCK_TASK_TEST_GRID,
)


def check_descriptive_stats(task):
    result_stat = task.calculate_descriptive_statistics()
    # remove descriptive task file
    task.metadata.descriptive_stat_path.unlink()
    print(task.metadata.name)
    print(result_stat)
    task_stat = task.expected_stats

    for key, value in result_stat.items():
        assert key in task_stat
        assert value == task_stat[key]


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
def test_descriptive_statistics_mock_tasks(task):
    check_descriptive_stats(task)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID)
def test_descriptive_statistics_mock_mieb_tasks(task):
    pytest.importorskip("PIL", reason="Image dependencies are not installed")
    check_descriptive_stats(task)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID)
def test_descriptive_statistics_mock_maeb_tasks(task):
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    check_descriptive_stats(task)


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
        TaskMetadata(
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


def test_task_hf_config():
    task = mteb.get_task("ArguAna")
    config = task._create_task_hf_config()
    assert config.name == task.metadata.name
    assert config.description == task.metadata.description
    assert config.evaluation_framework == "mteb"
    assert len(config.tasks) == 2

    assert config.tasks[0].id == "ArguAna"
    assert config.tasks[0].config is None

    assert config.tasks[1].id == "ArguAna_default_test"
    assert config.tasks[1].config == "default"
    assert config.tasks[1].split == "test"


def test_task_hf_config_from_existing():
    task1 = mteb.get_task("MIRACLRetrievalHardNegatives")
    task2 = mteb.get_task("MIRACLRetrievalHardNegatives.v2")

    config1 = task1._create_task_hf_config()
    config2 = task2._create_task_hf_config(config1)

    assert len(config2.tasks) == 2 * len(config1.tasks)

    assert any(t.id == "MIRACLRetrievalHardNegatives" for t in config2.tasks)
    assert any(t.id == "MIRACLRetrievalHardNegatives.v2" for t in config2.tasks)
