import logging

import pytest

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.get_tasks import get_tasks
from mteb.tasks import *
from mteb.tasks.historic_datasets import HISTORIC_DATASETS


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
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        license=None,
        task_subtypes=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="",
        avg_character_length=None,
        n_samples=None,
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
            category="s2s",
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            form=None,
            domains=None,
            license=None,
            task_subtypes=None,
            socioeconomic_status=None,
            annotations_creators=None,
            dialect=None,
            text_creation=None,
            bibtex_citation="",
            avg_character_length=None,
            n_samples=None,
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
            category="s2s",
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            form=None,
            domains=None,
            license=None,
            task_subtypes=None,
            socioeconomic_status=None,
            annotations_creators=None,
            dialect=None,
            text_creation=None,
            bibtex_citation="",
            avg_character_length=None,
            n_samples=None,
        )


def test_given_none_revision_path_then_it_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        my_task = TaskMetadata(
            name="MyTask",
            dataset={"path": "test/dataset", "revision": None},
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            form=None,
            domains=None,
            license=None,
            task_subtypes=None,
            socioeconomic_status=None,
            annotations_creators=None,
            dialect=None,
            text_creation=None,
            bibtex_citation="",
            avg_character_length=None,
            n_samples=None,
        )

        assert my_task.dataset["revision"] is None

        warning_logs = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warning_logs) == 1
        assert (
            warning_logs[0].message == "Revision missing for the dataset test/dataset. "
            "It is encourage to specify a dataset revision for reproducability."
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
            category="s2s",
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            form=None,
            domains=None,
            license=None,
            task_subtypes=None,
            socioeconomic_status=None,
            annotations_creators=None,
            dialect=None,
            text_creation=None,
            bibtex_citation="",
            avg_character_length=None,
            n_samples=None,
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
            category="s2s",
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=("2021-01-01", "2021-12-31"),
            form=["written"],
            domains=["Non-fiction"],
            license="mit",
            task_subtypes=["Thematic clustering"],
            socioeconomic_status="high",
            annotations_creators="expert-annotated",
            dialect=None,
            text_creation="found",
            bibtex_citation="Someone et al",
            avg_character_length={"train": 1},
            n_samples={"train": 1},
        ).is_filled()
        is True
    )


def test_all_metadata_is_filled():
    all_tasks = get_tasks()

    unfilled_metadata = []
    for task in all_tasks:
        if task.metadata.name not in HISTORIC_DATASETS:
            if not task.metadata.is_filled():
                unfilled_metadata.append(task.metadata.name)
    if unfilled_metadata:
        raise ValueError(
            f"The metadata of the following datasets is not filled: {unfilled_metadata}"
        )
