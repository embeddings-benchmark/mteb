from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union
from unittest.mock import Mock, patch

import aiohttp
import pytest
import srsly
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask, CrosslingualTask, MultilingualTask, TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.tasks.BitextMining.da.BornholmskBitextMining import BornholmBitextMining

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MTEB().tasks_cls)
@patch("datasets.load_dataset")
def test_load_data(mock_load_dataset: Mock, task: AbsTask):
    # TODO: We skip because this load_data is completely different.
    if isinstance(task, AbsTaskRetrieval):
        pytest.skip()
    with patch.object(task, "dataset_transform") as mock_dataset_transform:
        task.load_data()
        mock_load_dataset.assert_called()

        # They don't yet but should they so they can be expanded more easily?
        if not task.is_crosslingual and not task.is_multilingual:
            mock_dataset_transform.assert_called_once()


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


@dataclass(frozen=True)
class HFSpecificationError(Exception):
    name: str
    error: str


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


# asyncio.to_thread is only available in Python 3.9
# Have copied the source code. If deprecating support for 3.8, use the built-in function.
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


async def check_hf_lang_configuration_is_valid(
    metadata: TaskMetadata, hf_name: str | None
) -> Sequence[HFSpecificationError]:
    errors: list[HFSpecificationError] = []
    kw_args = copy.deepcopy(
        metadata.dataset
    )  # Avoid mutating it so you invalidate the cache
    kw_args["streaming"] = True
    kw_args["task_name"] = metadata.name
    if hf_name is not None:
        kw_args["name"] = hf_name

    try:
        ds = await to_thread(
            load_dataset,
            **kw_args,
        )
    except Exception as e:
        return [HFSpecificationError(metadata.name, str(e))]

    for split in metadata.eval_splits:
        if split not in ds.keys():  # type: ignore
            errors.append(
                HFSpecificationError(metadata.name, f"Split {split} not found")
            )

    return errors


async def check_task_hf_specification(
    task: AbsTask,
) -> Sequence[HFSpecificationError]:
    # Check if task has been already validated
    # Add the task name to the dataset metadata so it becomes part of cache invalidation
    # E.g. some tasks have the same dataset but different splits, so we want to check them independently
    task.metadata.dataset["task_name"] = task.metadata.name
    cache_path = Path(__file__).parent / "checked_hf_specifications.jsonl"
    if task.metadata.dataset in list(srsly.read_jsonl(cache_path)):
        return []

    logging.info(f"Checking task {task.metadata.name}")
    metadata = task.metadata

    # The meaning of an eval_lang differs depending on the task.
    # For most tasks, it is purely metadata, with no functional effect on the dataset.
    # For MultilingualTask and CrosslingualTask, it is used to load the dataset.
    if isinstance(task, (MultilingualTask, CrosslingualTask)):
        hf_names = [hf_name for hf_name in metadata.eval_langs]
    elif isinstance(task, AbsTaskRetrieval):
        hf_names = ["corpus", "queries"]
    else:
        # Some datasets specify the name of the dataset in the metadata. Use that if it exists.
        hf_names = [task.metadata.dataset.get("name", None)]

    errors = [
        await check_hf_lang_configuration_is_valid(metadata=metadata, hf_name=name)
        for name in hf_names
    ]

    # Add to the cache
    srsly.write_jsonl(
        cache_path, [task.metadata.dataset], append=True, append_new_line=False
    )
    return [error for task_errors in errors for error in task_errors]


async def check_dataset_spec_is_valid_on_hf(
    tasks: Sequence[AbsTask],
) -> Sequence[HFSpecificationError]:
    errors = await asyncio.gather(
        *[check_task_hf_specification(task) for task in tasks]
    )
    return [error for task_errors in errors for error in task_errors]


def test_dataset_conforms_to_schema():
    tasks = MTEB().tasks_cls
    selected_tasks = [
        task
        for task in tasks
        if not any(
            name in task.metadata.name
            for name in [
                "DiaBla",
                "AI-Sweden/SuperLim",
                "FloresBitextMining",
                "FloresClusteringS2S",
                "OpusparcusPC",
                "GerDaLIR",
                "GermanQuAD-Retrieval",
                "SpanishPassageRetrievalS2P",
                "SpanishPassageRetrievalS2S",
                "SyntecRetrieval",
                "AlloprofRetrieval",
                "XMarket",
                "NorQuadRetrieval",
                "ArguAna-PL",
                "MSMARCOv2",
                "FiQA-PL",
                "NarrativeQARetrieval",
                "SCIDOCS-PL",
                "NFCorpus",
                "SciFact-PL",
                "MSMARCO-PL",
                "HotpotQA-PL",
                "VideoRetrieval",
                "T2Retrieval",
                "MMarcoRetrieval",
                "MedicalRetrieval",
                "DBPedia-PL",
                "TRECCOVID-PL",
                "NQ-PL",
                "CovidRetrieval",
                "CmedqaRetrieval",
                "HotpotQA",
                "DuRetrieval",
                "EcomRetrieval",
                "Ko-mrtydi",
                "Quora-PL",
                "NFCorpus-PL",
                "MLSUMClusteringP2P",
                "SwednClustering",
                "SwednRetrieval",
                "NordicLangClassification",
                "SweFaqRetrieval",
                "MLSUMClusteringS2S",
                "DalajClassification",
            ]
        )
    ]
    errors = asyncio.run(check_dataset_spec_is_valid_on_hf(selected_tasks))
    if errors:
        pretty_print = "\n".join([f"{err.name} - {err.error}" for err in errors])
        assert False, f"Datasets not available on Hugging Face:\n{pretty_print}"
