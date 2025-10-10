"""Tests for testing the load_data method of all tasks"""
# TODO: KCE I would probably delete this test as it really doesn't test a lot

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.overview import get_tasks

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
