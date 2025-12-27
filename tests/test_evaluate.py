import logging
from copy import copy
from pathlib import Path

import pytest
from datasets.exceptions import DatasetNotFoundError

import mteb
from mteb.abstasks.abstask import AbsTask
from mteb.cache import ResultCache
from mteb.models.models_protocols import EncoderProtocol
from tests.mock_models import MockSentenceTransformer
from tests.mock_tasks import (
    MockAggregatedTask,
    MockClassificationTask,
    MockMultilingualRetrievalTask,
    MockRetrievalTask,
)

mock_classification = (MockSentenceTransformer(), MockClassificationTask(), 0.5)
mock_retrieval = (
    MockSentenceTransformer(),
    MockRetrievalTask(),
    pytest.approx(0.63093),
)


@pytest.mark.parametrize(
    "model, task, expected_score",
    [mock_classification, mock_retrieval],
    ids=["mock_classification", "mock_retrieval"],
)
def test_evaluate(model: EncoderProtocol, task: AbsTask, expected_score: float):
    results = mteb.evaluate(model, task, cache=None, co2_tracker=False)

    assert len(results) == 1, "should return exactly one result"
    result = results[0]

    assert result.task_name == task.metadata.name, "results should match the task"
    assert set(result.eval_splits) == set(task.eval_splits), "splits should match task."
    assert result.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize(
    "model, tasks",
    [(MockSentenceTransformer(), [MockClassificationTask(), MockRetrievalTask()])],
    ids=["mock_clf_and_retrieval"],
)
def test_evaluate_w_multiple_tasks(model: EncoderProtocol, tasks: list[AbsTask]):
    results = mteb.evaluate(model, tasks, cache=None, co2_tracker=False)
    assert len(results) == len(tasks), "should return exactly one result per task"


@pytest.mark.parametrize(
    "model, task, expected_score", [mock_classification], ids=["mock_classification"]
)
def test_evaluate_with_cache(
    model: EncoderProtocol, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)
    results = mteb.evaluate(model, task, cache=cache, co2_tracker=False)

    # Check if the cache is created
    path = cache.get_task_result_path(
        task.metadata.name,
        results.model_name.replace("/", "__"),
        results.model_revision,
    )
    model_meta_path = path.parent / "model_meta.json"
    assert path.exists() and path.is_file(), "cache file should exist"
    assert path.suffix == ".json", "cache file should be a json file"
    assert model_meta_path.exists(), "no model meta path is saved"

    result = results[0]
    assert result.task_name == task.metadata.name, "results should match the task"
    assert set(result.eval_splits) == set(task.eval_splits), "splits should match task."
    assert result.get_score() == expected_score, (
        "main score should match the expected value"
    )

    # test cache re-use
    cached_results = mteb.evaluate(
        model, task, cache=cache, overwrite_strategy="only-cache"
    )
    cached_result = cached_results[0]
    assert cached_result.task_name == task.metadata.name, (
        "results should match the task"
    )
    assert cached_result.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize(
    "model, task, expected_score,splits",
    [
        (MockSentenceTransformer(), MockClassificationTask(), 0.5, ["train"])
    ],  # default split is "test" so this will run "train" and then ["test", "train"], also means that expected score can be different
    ids=["mock_classification"],
)
def test_evaluate_w_missing_splits(
    model: EncoderProtocol,
    task: AbsTask,
    expected_score: float,
    splits: list[str],
    tmp_path: Path,
):
    cache = ResultCache(tmp_path)

    task._eval_splits = splits
    # ^ eq. to mteb.get_task("name", splits=[...])

    results = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
    result = results[0]

    assert set(result.eval_splits) == set(splits)
    task._eval_splits = list(set(list(task.metadata.eval_splits) + splits))
    results = mteb.evaluate(model, task, cache=cache)
    updated = results[0]

    assert set(updated.eval_splits) != set(result.eval_splits)
    assert set(updated.eval_splits).issuperset(set(task.metadata.eval_splits))

    assert updated.task_name == task.metadata.name, "results should match the task"
    assert updated.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize(
    "task", [MockClassificationTask()], ids=["mock_classification"]
)
def test_cache_hit(task: AbsTask):
    """Test that evaluating with 'only-cache' raises an error when there are no cache hit."""
    model = mteb.get_model("baseline/random-encoder-baseline")
    with pytest.raises(
        ValueError,
        match="overwrite_strategy is set to 'only-cache' and the results file exists",
    ):
        mteb.evaluate(model, task, overwrite_strategy="only-cache")


@pytest.mark.parametrize(
    "model, task, expected_score",
    [(MockSentenceTransformer(), MockMultilingualRetrievalTask(), 0.63093)],
    ids=["mock_retrieval"],
)
def test_evaluate_w_missing_subset(
    model: EncoderProtocol, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)

    hf_subsets = list(task.hf_subsets)
    task.hf_subsets = hf_subsets[:-1]
    # ^ eq. to mteb.get_task("name", splits=[...])

    results = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
    result = results[0]

    assert set(result.hf_subsets) == set(hf_subsets[:1])

    task.hf_subsets = hf_subsets
    results = mteb.evaluate(model, task, cache=cache)
    updated = results[0]

    assert set(updated.hf_subsets) != set(result.hf_subsets)
    assert set(updated.hf_subsets) == set(task.metadata.hf_subsets)

    assert updated.task_name == task.metadata.name, "results should match the task"
    assert updated.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize(
    "model, task, expected_score, splits",
    [(*mock_classification, ["train"])],
    ids=["mock_classification"],
)
def test_evaluate_overwrites(
    model: EncoderProtocol,
    task: AbsTask,
    expected_score: float,
    splits: list[str],
    tmp_path: Path,
):
    cache = ResultCache(tmp_path)

    task._eval_splits = splits
    # ^ eq. to mteb.get_task("name", splits=[...])

    # run part of a task to make an incomplete result
    results = mteb.evaluate(model, task, cache=cache, co2_tracker=False)

    task._eval_splits = task.metadata.eval_splits  # reset splits to default

    with pytest.raises(ValueError):
        results = mteb.evaluate(
            model, task, cache=cache, overwrite_strategy="only-cache"
        )

    with pytest.raises(ValueError):
        results = mteb.evaluate(model, task, cache=cache, overwrite_strategy="never")

    # should just overwrite
    results = mteb.evaluate(model, task, cache=cache, overwrite_strategy="always")
    assert results[0].get_score() == expected_score, (
        "main score should match the expected value"
    )


def test_evaluate_aggregated_task():
    model = mteb.get_model("baseline/random-encoder-baseline")
    task = MockAggregatedTask()
    mteb.evaluate(model, task, cache=None)


def test_run_private_task_warning(caplog):
    """Test that a warning is correctly logged in an attempt run a private dataset is made"""
    task = mteb.get_task("Code1Retrieval")

    def load_data_dataset_not_found():
        raise DatasetNotFoundError

    task.load_data = load_data_dataset_not_found
    model = mteb.get_model("baseline/random-encoder-baseline")

    with caplog.at_level(logging.WARNING):
        result = mteb.evaluate(model, task, cache=None)
        assert len(result.task_results) == 0
        assert "Dataset for private task 'Code1Retrieval' not found" in caplog.text


def test_run_private_task():
    """Tests that private task is run if it is possible to load the data"""
    task = MockRetrievalTask()
    task_metadata = copy(task.metadata)
    task_metadata.is_public = False
    task.metadata = task_metadata
    model = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(model, task, cache=None, public_only=False)
    assert len(results.task_results) == 1


def test_run_task_raise_error():
    """Test that the error is not caught unintentionally"""
    task = MockRetrievalTask()

    def load_error():
        raise RuntimeError("Test error")

    task.load_data = load_error
    model = mteb.get_model("baseline/random-encoder-baseline")
    with pytest.raises(RuntimeError, match="Test error"):
        mteb.evaluate(model, task, cache=None)


def test_run_list_with_error():
    """Test that errors are correctly suppressed, when specified"""
    error_task = MockRetrievalTask()

    def load_error():
        raise RuntimeError("Test error")

    error_task.load_data = load_error
    task = MockRetrievalTask()

    model = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(model, [error_task, task], cache=None, raise_error=False)
    assert len(results.task_results) == 1
    assert len(results.exceptions) == 1
