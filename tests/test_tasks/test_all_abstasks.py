from __future__ import annotations

import logging
import os
from unittest.mock import Mock, patch

import huggingface_hub
import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.overview import _TASKS_REGISTRY, get_tasks

from ..test_benchmark.task_grid import (
    MOCK_MIEB_TASK_GRID_AS_STRING,
    MOCK_TASK_TEST_GRID_AS_STRING,
)

logging.basicConfig(level=logging.INFO)

ALL_MOCK_TASKS = MOCK_TASK_TEST_GRID_AS_STRING + MOCK_MIEB_TASK_GRID_AS_STRING

tasks = [
    t
    for t in get_tasks(exclude_superseded=False)
    if t.metadata.name not in ALL_MOCK_TASKS
]


datasets_not_available = [
    "AfriSentiLangClassification",
]


_original_dataset_revisions = list(
    {  # deduplicate as multiple tasks rely on the same dataset (save us at least 100 test cases)
        (t.metadata.dataset["path"], t.metadata.dataset["revision"])
        for t in mteb.get_tasks(exclude_superseded=False)
        if not isinstance(t, AbsTaskAggregate)
        and t.metadata.name not in datasets_not_available
        and t.metadata.name not in ALL_MOCK_TASKS
    }
)

custom_revisions = os.getenv("CUSTOM_DATASET_REVISIONS")
if custom_revisions:
    # Parse comma-separated list of "path:revision" pairs
    dataset_revisions = [
        tuple(pair.split(":", 1)) for pair in custom_revisions.split(",") if ":" in pair
    ]
else:
    dataset_revisions = _original_dataset_revisions


@pytest.mark.parametrize("task", tasks)
@patch("datasets.load_dataset")
@patch("datasets.concatenate_datasets")
def test_load_data(
    mock_concatenate_datasets: Mock, mock_load_dataset: Mock, task: AbsTask
):
    # TODO: We skip because this load_data is completely different.
    if (
        isinstance(task, AbsTaskRetrieval)
        or isinstance(task, AbsTaskImageTextPairClassification)
        or task.metadata.is_multilingual
    ):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.metadata.is_multilingual:
            mock_dataset_transform.assert_called_once()


@pytest.mark.test_datasets
@pytest.mark.flaky(
    reruns=5,
    reruns_delay=12,
    only_rerun=["AssertionError"],
    reason="May fail due to network issues",
)
@pytest.mark.parametrize("dataset_revision", dataset_revisions)
def test_dataset_on_hf(dataset_revision: tuple[str, str]):
    repo_id, revision = dataset_revision
    try:
        huggingface_hub.dataset_info(repo_id, revision=revision)
    except (
        huggingface_hub.errors.RepositoryNotFoundError,
        huggingface_hub.errors.RevisionNotFoundError,
    ):
        assert False, f"Dataset {repo_id} - {revision} not available"
    except Exception as e:
        assert False, f"Dataset {repo_id} - {revision} failed with {e}"


def test_superseded_dataset_exists():
    tasks = mteb.get_tasks(exclude_superseded=False)
    for task in tasks:
        if task.superseded_by:
            assert task.superseded_by in _TASKS_REGISTRY, (
                f"{task} is superseded by {task.superseded_by} but {task.superseded_by} is not in the TASKS_REGISTRY"
            )


def test_is_aggregate_property_correct():
    tasks = mteb.get_tasks()

    for task in tasks:
        assert task.is_aggregate == isinstance(task, AbsTaskAggregate)
