from importlib.metadata import version
from pathlib import Path

import pytest

import mteb
from mteb import ResultCache
from mteb._hf_integration.eval_result_model import (
    HFEvalResultDataset,  # noqa: PLC2701
)
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.results import TaskResult

tests_folder = Path(__file__).parent.parent


class DummyTask(AbsTask):
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
        superseded_by="newer_task",
    )

    def evaluate(self, model, split: str = "test"):
        pass

    def _evaluate_subset(self, **kwargs):
        pass

    def _calculate_descriptive_statistics_from_split(  # noqa: PLR6301
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
        task=DummyTask(),
        scores=scores,
        evaluation_time=evaluation_time,
    )


def test_task_results_get_score(task_result: TaskResult):
    """Test TaskResult class (this is the same as the example in the docstring)"""
    assert task_result.get_score() == 0.55
    assert task_result.get_score(languages=["eng"]) == 0.55
    assert task_result.get_score(languages=["fra"]) == 0.6


def test_task_results_to_dict(task_result: TaskResult):
    mteb_ver = version("mteb")
    dict_repr = {
        "dataset_revision": "1.0",
        "task_name": "dummy_task",
        "mteb_version": mteb_ver,
        "evaluation_time": 100,
        "date": None,
        "kg_co2_emissions": None,
        "scores": {
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                    "mteb_version": mteb_ver,
                },
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                    "mteb_version": mteb_ver,
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


def test_per_subset_mteb_version(task_result: TaskResult):
    """Test that each subset score dict includes mteb_version."""
    mteb_ver = version("mteb")
    for split_scores in task_result.scores.values():
        for subset_scores in split_scores:
            assert "mteb_version" in subset_scores
            assert subset_scores["mteb_version"] == mteb_ver


def test_merge_across_mteb_versions():
    """Test that results from different MTEB versions can be merged."""
    existing = TaskResult(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="2.12.4",
        scores={
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                    "mteb_version": "2.12.4",
                },
            ]
        },
        evaluation_time=50,
    )

    new = TaskResult(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="2.20.1",
        scores={
            "train": [
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                    "mteb_version": "2.20.1",
                },
            ]
        },
        evaluation_time=60,
    )

    assert existing.is_mergeable(new)
    merged = existing.merge(new)

    # Both subsets should be present
    assert len(merged.scores["train"]) == 2
    subsets = {s["hf_subset"]: s for s in merged.scores["train"]}
    assert subsets["en-de"]["mteb_version"] == "2.12.4"
    assert subsets["en-fr"]["mteb_version"] == "2.20.1"

    # Top-level version should be a range when subsets differ
    assert merged.mteb_version == "2.12.4-2.20.1"


def test_merge_without_per_subset_version():
    """Test merging old results that lack per-subset mteb_version."""
    existing = TaskResult(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="2.12.4",
        scores={
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                },
            ]
        },
        evaluation_time=50,
    )

    new = TaskResult(
        dataset_revision="1.0",
        task_name="dummy_task",
        mteb_version="2.20.1",
        scores={
            "train": [
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                    "mteb_version": "2.20.1",
                },
            ]
        },
        evaluation_time=60,
    )

    merged = existing.merge(new)
    # Old subset has no per-subset version, new one does
    subsets = {s["hf_subset"]: s for s in merged.scores["train"]}
    assert "mteb_version" not in subsets["en-de"]
    assert subsets["en-fr"]["mteb_version"] == "2.20.1"

    # Top-level should be the latest found across subsets
    assert merged.mteb_version == "2.20.1"


@pytest.mark.parametrize(
    "path", list((tests_folder / "historic_results").glob("*.json"))
)
def test_mteb_results_from_historic(path: Path):
    mteb_result = TaskResult.from_disk(path, load_historic_data=True)
    assert isinstance(mteb_result, TaskResult)


def test_to_hf_result(mock_mteb_cache: ResultCache):
    task_name = "Banking77Classification"
    task_metadata = mteb.get_task(task_name).metadata
    benchmark_result = mock_mteb_cache.load_results(
        models=["mteb/baseline-random-encoder"], tasks=[task_name]
    )
    user_name = "test_user"
    task_result = benchmark_result.model_results[0].task_results[0]
    hf_results = task_result._to_hf_benchmark_result(user_name)
    assert len(hf_results) == 2

    assert hf_results[-1].dataset.task_id == task_name
    assert hf_results[0].dataset == HFEvalResultDataset(
        id=task_metadata.dataset["path"],
        task_id=f"{task_name}_default_test",
        revision=task_metadata.revision,
    )

    hf_result = hf_results[-1]
    assert hf_result.dataset == HFEvalResultDataset(
        id=task_metadata.dataset["path"],
        task_id=task_name,
        revision=task_metadata.revision,
    )
    assert hf_result.value == 1.2532
    assert hf_result.source.user == user_name

    assert (
        hf_results.to_yaml()
        == """- dataset:
    id: mteb/banking77
    task_id: Banking77Classification_default_test
    revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
  value: 1.2532
  notes: Obtained using MTEB v2.4.2
  source:
    url: https://github.com/embeddings-benchmark/mteb/
    name: Obtained using MTEB v2.4.2
    user: test_user
- dataset:
    id: mteb/banking77
    task_id: Banking77Classification
    revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
  value: 1.2532
  notes: Obtained using MTEB v2.4.2
  source:
    url: https://github.com/embeddings-benchmark/mteb/
    name: Obtained using MTEB v2.4.2
    user: test_user
"""
    )
