from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest
import requests

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSpeedTask import AbsTaskSpeedTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultiSubsetLoader import MultiSubsetLoader
from mteb.overview import TASKS_REGISTRY

from ..test_benchmark.task_grid import (
    MOCK_MIEB_TASK_GRID_AS_STRING,
    MOCK_TASK_TEST_GRID_AS_STRING,
)

logging.basicConfig(level=logging.INFO)

ALL_MOCK_TASKS = MOCK_TASK_TEST_GRID_AS_STRING + MOCK_MIEB_TASK_GRID_AS_STRING

tasks = [t for t in MTEB().tasks_cls if t.metadata.name not in ALL_MOCK_TASKS]


dataset_revisions = list(
    {  # deduplicate as multiple tasks rely on the same dataset (save us at least 100 test cases)
        (t.metadata.dataset["path"], t.metadata.dataset["revision"])
        for t in mteb.get_tasks(exclude_superseded=False)
        if not isinstance(t, (AbsTaskAggregate, AbsTaskSpeedTask))
        and t.metadata.name != "AfriSentiLangClassification"
        and t.metadata.name not in ALL_MOCK_TASKS
    }
)


@pytest.mark.parametrize("task", tasks)
@patch("datasets.load_dataset")
@patch("datasets.concatenate_datasets")
def test_load_data(
    mock_concatenate_datasets: Mock, mock_load_dataset: Mock, task: AbsTask
):
    # TODO: We skip because this load_data is completely different.
    if (
        isinstance(task, AbsTaskRetrieval)
        or isinstance(task, AbsTaskAny2AnyRetrieval)
        or isinstance(task, AbsTaskInstructionRetrieval)
        or isinstance(task, MultiSubsetLoader)
        or isinstance(task, AbsTaskSpeedTask)
        or isinstance(task, AbsTaskAny2AnyMultiChoice)
    ):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.is_multilingual:
            mock_dataset_transform.assert_called_once()


@pytest.mark.flaky(
    reruns=3,
    reruns_delay=5,
    only_rerun=["AssertionError"],
    reason="May fail due to network issues",
)
@pytest.mark.parametrize("dataset_revision", dataset_revisions)
def test_dataset_on_hf(dataset_revision: tuple[str, str]):
    dataset, revision = dataset_revision
    url = f"https://huggingface.co/datasets/{dataset}/tree/{revision}"
    response = requests.head(url)

    assert response.status_code == 200, (
        f"Dataset {dataset} - {revision} not available. Status code: {response.status_code}"
    )


def test_superseded_dataset_exists():
    tasks = mteb.get_tasks(exclude_superseded=False)
    for task in tasks:
        if task.superseded_by:
            assert task.superseded_by in TASKS_REGISTRY, (
                f"{task} is superseded by {task.superseded_by} but {task.superseded_by} is not in the TASKS_REGISTRY"
            )
