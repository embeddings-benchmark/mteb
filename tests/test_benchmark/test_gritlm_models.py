from typing import Union

import pytest
import torch

import mteb
from mteb.models.instructions import task_to_instruction
from tests.test_benchmark.task_grid import MOCK_TASK_TEST_GRID

pytest.importorskip("gritlm")
from gritlm import GritLM


class MockGritLMWrapper(GritLM):
    def encode(self, *args, **kwargs):
        if "prompt_name" in kwargs:
            if "instruction" in kwargs:
                raise ValueError(
                    "Cannot specify both `prompt_name` and `instruction`."
                )
        return torch.randn(len(args[0]), 10)

    def encode_corpus(self, *args, **kwargs):
        kwargs["is_query"] = False
        return torch.randn(len(args[0]), 10)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize(
    "model", [MockGritLMWrapper()]
)
def test_benchmark_encoders_on_task(
    task: Union[str, mteb.AbsTask], model: mteb.Encoder
):
    """Test that a task can be fetched and run using a variety of encoders"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder="tests/results", overwrite_results=True)


def test_get_prompt():
    assert task_to_instruction("CovidRetrieval") == "Given a question on COVID-19, retrieve news articles that answer the question"
    assert task_to_instruction("CovidRetrieval", is_query=False) == ""