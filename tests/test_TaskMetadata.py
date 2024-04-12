import logging

import pytest

from mteb.abstasks.TaskMetadata import TaskMetadata


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
        bibtex_citation=None,
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
            bibtex_citation=None,
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
            bibtex_citation=None,
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
            bibtex_citation=None,
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
