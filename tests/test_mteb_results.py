from importlib.metadata import version

import mteb
from mteb import AbsTask
from mteb.abstasks.mteb_result import MTEBResults


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

    def evaluate(self, model, split: str = "test"):
        pass


def test_mteb_results():
    """Test MTEBResults class (this is the same as the example in the docstring)"""
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

    mteb_results = MTEBResults.from_task_results(task=DummyTask(), scores=scores)

    assert mteb_results.get_score() == 0.55
    assert mteb_results.get_score(languages=["eng"]) == 0.55
    assert mteb_results.get_score(languages=["fra"]) == 0.6
    dict_repr = {
        "dataset_revision": "1.0",
        "task_name": "dummy_task",
        "mteb_version": version("mteb"),
        "scores": {
            "train": [
                {
                    "main_score": 0.5,
                    "evaluation_time": 100,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                },
                {
                    "main_score": 0.6,
                    "evaluation_time": 200,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                },
            ]
        },
    }
    assert mteb_results.to_dict() == dict_repr
