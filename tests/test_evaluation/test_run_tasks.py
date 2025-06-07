from __future__ import annotations

from pathlib import Path

import pytest

import mteb
from mteb.abstasks.AbsTask import AbsTask
from mteb.cache import ResultCache
from mteb.encoder_interface import Encoder
from tests.test_benchmark.mock_models import MockSentenceTransformer
from tests.test_benchmark.mock_tasks import (
    MockMultilingualRetrievalTask,
    MockRetrievalTask,
)

simple_test_case = (MockSentenceTransformer(), MockRetrievalTask(), 0.63093)


@pytest.mark.parametrize("model, task, expected_score", [simple_test_case])
def test_run_tasks(model: Encoder, task: AbsTask, expected_score: float):
    results = mteb.run_tasks(model, task, cache=None)

    assert len(results) == 1, "should return exactly one result"
    result = results[0]

    assert result.task_name == task.metadata.name, "results should match the task"
    assert set(result.eval_splits) == set(task.eval_splits), "splits should match task."
    assert result.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize("model, task, expected_score", [simple_test_case])
def test_run_tasks_with_cache(
    model: Encoder, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)
    results = mteb.run_tasks(model, task, cache=cache)

    # Check if the cache is created
    path = cache.get_task_result_path(
        task.metadata.name,
        results.model_name.replace("/", "__"),
        results.model_revision,  # type: ignore
    )
    assert path.exists() and path.is_file(), "cache file should exist"
    assert path.suffix == ".json", "cache file should be a json file"

    result = results[0]
    assert result.task_name == task.metadata.name, "results should match the task"
    assert set(result.eval_splits) == set(task.eval_splits), "splits should match task."
    assert result.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize("model, task, expected_score", [simple_test_case])
def test_run_task_w_missing_splits(
    model: Encoder, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)

    all_splits = list(task.eval_splits)
    task._eval_splits = all_splits[:1]
    # ^ eq. to mteb.get_task("name", splits=[...])

    results = mteb.run_tasks(model, task, cache=cache)
    result = results[0]

    assert set(result.eval_splits) == set(all_splits[:1])

    task._eval_splits = all_splits
    results = mteb.run_tasks(model, task, cache=cache)
    updated = results[0]

    assert set(updated.eval_splits) != set(result.eval_splits)
    assert set(updated.eval_splits) == set(task.metadata.eval_splits)

    assert updated.task_name == task.metadata.name, "results should match the task"
    assert updated.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize(
    "model, task, expected_score",
    [
        (MockSentenceTransformer(), MockMultilingualRetrievalTask(), 0.63093),
    ],
)
def test_run_task_w_missing_subset(
    model: Encoder, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)

    hf_subsets = list(task.hf_subsets)
    task.hf_subsets = hf_subsets[:-1]
    # ^ eq. to mteb.get_task("name", splits=[...])

    results = mteb.run_tasks(model, task, cache=cache)
    result = results[0]

    assert set(result.hf_subsets) == set(hf_subsets[:1])

    task.hf_subsets = hf_subsets
    results = mteb.run_tasks(model, task, cache=cache)
    updated = results[0]

    assert set(updated.hf_subsets) != set(result.hf_subsets)
    assert set(updated.hf_subsets) == set(task.metadata.hf_subsets)

    assert updated.task_name == task.metadata.name, "results should match the task"
    assert updated.get_score() == expected_score, (
        "main score should match the expected value"
    )


@pytest.mark.parametrize("model, task, expected_score", [simple_test_case])
def test_run_task_overwrites(
    model: Encoder, task: AbsTask, expected_score: float, tmp_path: Path
):
    cache = ResultCache(tmp_path)

    all_splits = list(task.eval_splits)
    task._eval_splits = all_splits[:1]
    # ^ eq. to mteb.get_task("name", splits=[...])

    # run part of a task to make an incomplete result
    results = mteb.run_tasks(model, task, cache=cache)

    with pytest.raises(ValueError):
        results = mteb.run_tasks(
            model, task, cache=cache, overwrite_strategy="only-cache"
        )

    with pytest.raises(ValueError):
        results = mteb.run_tasks(model, task, cache=cache, overwrite_strategy="never")

    # should just overwrite
    results = mteb.run_tasks(model, task, cache=cache, overwrite_strategy="always")
    assert results[0].get_score() == expected_score, (
        "main score should match the expected value"
    )
