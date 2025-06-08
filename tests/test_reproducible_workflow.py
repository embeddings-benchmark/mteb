from __future__ import annotations

import logging

import pytest

import mteb
from mteb import MTEB
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from tests.test_benchmark.task_grid import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task_name", ["BornholmBitextMining"])
@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
@pytest.mark.parametrize("model_revision", ["8b3219a92973c328a8e22fadcfa821b5dc75636a"])
def test_reproducibility_workflow(task_name: str, model_name: str, model_revision: str):
    """Test that a model and a task can be fetched and run in a reproducible fashion."""
    model_meta = mteb.get_model_meta(model_name, revision=model_revision)
    task = mteb.get_task(task_name)

    assert isinstance(model_meta, ModelMeta)
    assert isinstance(task, mteb.AbsTask)

    model = mteb.get_model(model_name, revision=model_revision)
    assert isinstance(model, Encoder)

    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize(
    "task_name",
    TASK_TEST_GRID
    + [
        "BitextMining",
        "Classification",
        "MultilabelClassification",
        "Clustering",
        "PairClassification",
        "Reranking",
        "Retrieval",
        "STS",
        "Summarization",
        "InstructionRetrieval",
        "Speed",
    ],
)
def test_validate_task_to_prompt_name(task_name: str | mteb.AbsTask):
    if isinstance(task_name, mteb.AbsTask):
        task_names = [task_name.metadata.name]
    else:
        task_names = [task_name]

    model_prompts = {task_name: "prompt_name" for task_name in task_names}
    model_prompts |= {task_name + "-query": "prompt_name" for task_name in task_names}
    model_prompts |= {task_name + "-passage": "prompt_name" for task_name in task_names}
    model_prompts |= {
        "query": "prompt_name",
        "passage": "prompt_name",
    }
    Wrapper.validate_task_to_prompt_name(model_prompts)


def test_validate_task_to_prompt_name_fail():
    with pytest.raises(KeyError):
        Wrapper.validate_task_to_prompt_name(
            {"task_name": "prompt_name", "task_name-query": "prompt_name"}
        )

    with pytest.raises(KeyError):
        Wrapper.validate_task_to_prompt_name({"task_name-task_name": "prompt_name"})
