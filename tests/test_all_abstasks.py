from __future__ import annotations

import asyncio
import logging
from typing import Union

import aiohttp
import pytest
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.tasks.BitextMining.da.BornholmskBitextMining import BornholmBitextMining

logging.basicConfig(level=logging.INFO)


def test_two_mteb_tasks():
    """
    Test that two tasks can be fetched and run
    """
    model = SentenceTransformer("average_word_embeddings_komninos")
    eval = MTEB(
        tasks=[
            "STS12",
            "SummEval",
        ]
    )
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize(
    "task",
    [
        BornholmBitextMining(),
        "TwentyNewsgroupsClustering",
        "Banking77Classification",
        "SciDocsRR",
        "SprintDuplicateQuestions",
        "NFCorpus",
        "STS12",
        "SummEval",
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "average_word_embeddings_levy_dependency",
    ],
)
def test_mteb_task(task: Union[str, AbsTask], model_name: str):
    """
    Test that a task can be fetched and run
    """
    model = SentenceTransformer(model_name)
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)


def test_all_tasks_fetch():
    """
    Test that all tasks can be fetched
    """
    MTEB.mteb_tasks()


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
                session, task.metadata.hf_hub_name, task.metadata.revision
            )
            for task in tasks
        ]
        datasets_exists = await asyncio.gather(*tasks_checks)

    for task, ds_exists in zip(tasks, datasets_exists):
        if not ds_exists:
            does_not_exist.append((task.metadata.hf_hub_name, task.metadata.revision))

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
