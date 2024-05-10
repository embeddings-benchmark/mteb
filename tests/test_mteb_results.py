import mteb
from mteb import AbsTask
from mteb.abstasks.mteb_result import MTEBResults
from mteb.overview import TASKS_REGISTRY


class DummyTask(AbsTask):
    superseeded_by = "newer_task"
    metadata = mteb.TaskMetadata(
        name="dummy_task",
        description="dummy task for testing",
        dataset={"revision": "1.0", "path": "dummy_dataset"},
        type="Classification",
        category="p2p",
        eval_langs={
            "en-de": ["eng-Latn", "deu-Latn"],
            "en-fr": ["eng-Latn", "fra-Latn"],
        },
        main_score="main_score",
        eval_splits=["train"],
        domains=[],
        text_creation="created",
        reference="https://www.noreference.com",
        date=("2024-05-02", "2024-05-03"),
        form=[],
        task_subtypes=[],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        avg_character_length={},
        n_samples={},
    )


def test_mteb_results():
    """Test MTEBResults class (this is the same as the example in the docstring)"""
    TASKS_REGISTRY["dummy_task"] = DummyTask

    scores = {
        "train": {
            "en-de": {
                "main_score": 0.5,
                "evaluation_time": 100,
            },
            "en-fr": {
                "main_score": 0.6,
                "evaluation_time": 200,
            },
        },
    }

    mteb_results = MTEBResults(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="1.0",
        scores=scores,
    )

    assert mteb_results.get_main_score() == 0.55
    assert mteb_results.get_main_score(languages=["eng"]) == 0.55
    assert mteb_results.get_main_score(languages=["fra"]) == 0.6

    del TASKS_REGISTRY["dummy_task"]
