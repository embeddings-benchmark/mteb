from __future__ import annotations

import asyncio
import logging
from unittest.mock import Mock, patch

import aiohttp
import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSpeedTask import AbsTaskSpeedTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.overview import TASKS_REGISTRY, get_tasks

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
        or isinstance(task, AbsTaskReranking)
        or isinstance(task, AbsTaskAny2AnyRetrieval)
        or isinstance(task, AbsTaskSpeedTask)
        or isinstance(task, AbsTaskAny2AnyMultiChoice)
        or task.metadata.is_multilingual
    ):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.metadata.is_multilingual:
            mock_dataset_transform.assert_called_once()


async def check_dataset_on_hf(
    session: aiohttp.ClientSession, dataset: str, revision: str
) -> bool:
    url = f"https://huggingface.co/datasets/{dataset}/tree/{revision}"
    async with session.head(url) as response:
        return response.status == 200


async def check_datasets_are_available_on_hf(tasks):
    does_not_exist = []
    async with aiohttp.ClientSession() as session:
        tasks_checks = [
            check_dataset_on_hf(
                session,
                task.metadata.dataset["path"],
                task.metadata.dataset["revision"],
            )
            for task in tasks
            if not isinstance(task, AbsTaskSpeedTask)
        ]
        datasets_exists = await asyncio.gather(*tasks_checks)

    for task, ds_exists in zip(tasks, datasets_exists):
        if not ds_exists:
            does_not_exist.append(
                (
                    task.metadata.name,
                    task.metadata.dataset["path"],
                    task.metadata.dataset["revision"],
                )
            )

    if does_not_exist:
        pretty_print = "\n".join(
            [
                f"Name: {ds[0]} - repo {ds[1]} - revision {ds[2]}"
                for ds in does_not_exist
            ]
        )
        assert False, f"Datasets not available on Hugging Face:\n{pretty_print}"


@pytest.mark.flaky(
    reruns=3,
    reruns_delay=5,
    only_rerun=["AssertionError"],
    reason="May fail due to network issues",
)
def test_dataset_availability():
    """Checks if the datasets are available on Hugging Face using both their name and revision."""
    tasks = get_tasks(exclude_superseded=False)
    tasks = [
        t
        for t in tasks
        # HOTFIX: Issue#1777. Remove this line when issue is resolved.
        if t.metadata.name != "AfriSentiLangClassification"
        # do not check aggregated tasks as they don't have a dataset
        and not isinstance(t, AbsTaskAggregate)
        and t.metadata.name not in ALL_MOCK_TASKS
    ]
    asyncio.run(check_datasets_are_available_on_hf(tasks))


def test_superseded_dataset_exists():
    tasks = mteb.get_tasks(exclude_superseded=False)
    for task in tasks:
        if task.superseded_by:
            assert task.superseded_by in TASKS_REGISTRY, (
                f"{task} is superseded by {task.superseded_by} but {task.superseded_by} is not in the TASKS_REGISTRY"
            )
