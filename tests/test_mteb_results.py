from importlib.metadata import version
from pathlib import Path

import pytest

import mteb
from mteb import AbsTask
from mteb.abstasks.MTEBResults import MTEBResults

tests_folder = Path(__file__).parent


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

    def _evaluate_subset(self, **kwargs):
        pass


def test_mteb_results():
    """Test MTEBResults class (this is the same as the example in the docstring)"""
    scores = {
        "train": {
            "en-de": {
                "main_score": 0.5,
            },
            "en-fr": {
                "main_score": 0.6,
            },
        },
    }

    evaluation_time = 100

    mteb_results = MTEBResults.from_task_results(
        task=DummyTask(), scores=scores, evaluation_time=evaluation_time
    )

    assert mteb_results.get_score() == 0.55
    assert mteb_results.get_score(languages=["eng"]) == 0.55
    assert mteb_results.get_score(languages=["fra"]) == 0.6
    dict_repr = {
        "dataset_revision": "1.0",
        "task_name": "dummy_task",
        "mteb_version": version("mteb"),
        "evaluation_time": 100,
        "kg_co2_emissions": None,
        "scores": {
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                },
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                },
            ]
        },
    }
    assert mteb_results.to_dict() == dict_repr


@pytest.mark.parametrize(
    "path", list((tests_folder / "historic_results").glob("*.json"))
)
def test_mteb_results_from_historic(path: Path):
    mteb_result = MTEBResults.from_disk(path, load_historic_data=True)
    assert isinstance(mteb_result, MTEBResults)
