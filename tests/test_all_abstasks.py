from __future__ import annotations

import asyncio
import logging
from unittest.mock import Mock, patch

import aiohttp
import pytest

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MTEB().tasks_cls)
@patch("datasets.load_dataset")
def test_load_data(mock_load_dataset: Mock, task: AbsTask):
    # TODO: We skip because this load_data is completely different.
    if isinstance(task, AbsTaskRetrieval) or isinstance(
        task, AbsTaskInstructionRetrieval
    ):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.is_crosslingual and not task.is_multilingual:
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
        ]
        datasets_exists = await asyncio.gather(*tasks_checks)

    for task, ds_exists in zip(tasks, datasets_exists):
        if not ds_exists:
            does_not_exist.append(
                (task.metadata.dataset["path"], task.metadata.dataset["revision"])
            )

    if does_not_exist:
        pretty_print = "\n".join(
            [f"{ds[0]} - revision {ds[1]}" for ds in does_not_exist]
        )
        assert False, f"Datasets not available on Hugging Face:\n{pretty_print}"


def test_dataset_availability():
    """
    Checks if the datasets are available on Hugging Face using both their name and revision.
    """
    tasks = MTEB().tasks_cls
    asyncio.run(check_datasets_are_available_on_hf(tasks))
