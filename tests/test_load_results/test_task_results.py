from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest

import mteb
from mteb import AbsTask
from mteb.load_results.task_results import TaskResult

tests_folder = Path(__file__).parent.parent


class DummyTask(AbsTask):
    superseded_by = "newer_task"
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
        return {}


@pytest.fixture()
def task_result():
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

    return TaskResult.from_task_results(
        task=DummyTask(), scores=scores, evaluation_time=evaluation_time
    )


def test_task_results_get_score(task_result: TaskResult):
    """Test TaskResult class (this is the same as the example in the docstring)"""
    assert task_result.get_score() == 0.55
    assert task_result.get_score(languages=["eng"]) == 0.55
    assert task_result.get_score(languages=["fra"]) == 0.6


def test_task_results_to_dict(task_result: TaskResult):
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
    assert task_result.to_dict() == dict_repr


def test_task_results_validate_and_filter():
    scores = {
        "train": {
            "en-de": {
                "main_score": 0.5,
            },
            "en-fr": {
                "main_score": 0.6,
            },
        },
        "test": {
            "en-de": {
                "main_score": 0.3,
            },
            "en-fr": {
                "main_score": 0.4,
            },
        },
    }

    evaluation_time = 100

    res = TaskResult.from_task_results(
        task=DummyTask(), scores=scores, evaluation_time=evaluation_time
    )

    task = DummyTask()
    task._eval_splits = ["train", "test"]
    res1 = res.validate_and_filter_scores(task=task)

    assert res1.scores.keys() == {"train", "test"}
    assert res1.get_score() == (0.5 + 0.6 + 0.3 + 0.4) / 4

    task._eval_splits = ["test"]
    res2 = res.validate_and_filter_scores(task=task)
    assert res2.scores.keys() == {"test"}
    assert res2.get_score() == (0.3 + 0.4) / 2  # only test scores

    task.hf_subsets = ["en-de"]
    task._eval_splits = ["train", "test"]
    res3 = res.validate_and_filter_scores(task=task)
    assert res3.scores.keys() == {"train", "test"}
    assert res3.get_score() == (0.5 + 0.3) / 2  # only en-de scores


@pytest.mark.parametrize(
    "path", list((tests_folder / "historic_results").glob("*.json"))
)
def test_mteb_results_from_historic(path: Path):
    mteb_result = TaskResult.from_disk(path, load_historic_data=True)
    assert isinstance(mteb_result, TaskResult)
