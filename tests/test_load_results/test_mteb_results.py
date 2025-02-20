from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest

from mteb.abstasks import AbsTask, TaskMetadata
from mteb.load_results.task_results import TaskResult

tests_folder = Path(__file__).parent.parent


class DummyTask(AbsTask):
    superseded_by = "newer_task"
    metadata = TaskMetadata(
        name="dummy_task",
        description="dummy task for testing",
        dataset={"revision": "1.0", "path": "dummy_dataset"},
        type="Classification",
        category="t2t",
        eval_langs={
            "en-de": ["eng-Latn", "deu-Latn"],
            "en-fr": ["eng-Latn", "fra-Latn"],
        },
        main_score="main_score",
        eval_splits=["train"],
        domains=[],
        reference="https://www.noreference.com",
        date=("2024-05-02", "2024-05-03"),
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        modalities=["text"],
        sample_creation="created",
    )

    def evaluate(self, model, split: str = "test"):
        pass

    def _evaluate_subset(self, **kwargs):
        pass

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall=False
    ) -> dict[str, float]:
        pass


def test_mteb_results():
    """Test TaskResult class (this is the same as the example in the docstring)"""
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

    mteb_results = TaskResult.from_task_results(
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
    mteb_result = TaskResult.from_disk(path, load_historic_data=True)
    assert isinstance(mteb_result, TaskResult)
