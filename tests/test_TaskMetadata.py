import pytest

from mteb.abstasks.TaskMetadata import TaskMetadata

def test_given_only_legacy_hf_hub_name_then_it_is_valid():
    my_task = TaskMetadata(
        name="MyTask",
        hf_hub_name="test/dataset",
        revision="1.0",
        description="testing",
        reference=None,
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
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


def test_given_only_dataset_then_it_is_valid():
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
        eval_langs=["en"],
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


def test_given_both_legacy_hf_hub_name_and_dataset_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(
            name="MyTask",
            hf_hub_name="test/dataset",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            eval_splits=["test"],
            eval_langs=["en"],
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


def test_given_missing_dataset_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(
            name="MyTask",
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            eval_splits=["test"],
            eval_langs=["en"],
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

