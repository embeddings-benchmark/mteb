"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

import mteb
from mteb.benchmarks import Benchmark

from .mock_models import MockNumpyEncoder, MockTorchbf16Encoder, MockTorchEncoder
from .task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("tasks", [MOCK_TASK_TEST_GRID[:2]])
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_mulitple_mteb_tasks(tasks: list[mteb.AbsTask], model: mteb.Encoder):
    """Test that multiple tasks can be run"""
    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize(
    "model", [MockNumpyEncoder(), MockTorchEncoder(), MockTorchbf16Encoder()]
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


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_prompt_name_passed_to_all_encodes(task_name: str | mteb.AbsTask):
    """Test that all tasks correctly pass down the task_name to the encoder which supports it, and that the encoder which does not support it does not
    receive it.
    """
    _task_name = (
        task_name.metadata.name if isinstance(task_name, mteb.AbsTask) else task_name
    )

    class MockEncoderWithInstructions(mteb.Encoder):
        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert prompt_name == _task_name
            return np.zeros((len(sentences), 10))

    class EncoderWithoutInstructions(SentenceTransformer):
        def encode(self, sentences, **kwargs):
            assert "prompt_name" not in kwargs
            return super().encode(sentences, **kwargs)

    if isinstance(task_name, mteb.AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockEncoderWithInstructions()
    eval.run(model, output_folder="tests/results", overwrite_results=True)
    # Test that the task_name is not passed down to the encoder
    model = EncoderWithoutInstructions("average_word_embeddings_levy_dependency")
    assert model.prompts == {}, "The encoder should not have any prompts"
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_encode_kwargs_passed_to_all_encodes(task_name: str | mteb.AbsTask):
    """Test that all tasks correctly pass down the encode_kwargs to the encoder."""
    my_encode_kwargs = {"no_one_uses_this_args": "but_its_here"}

    class MockEncoderWithKwargs(mteb.Encoder):
        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert kwargs == my_encode_kwargs
            return np.zeros((len(sentences), 10))

    if isinstance(task_name, mteb.AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockEncoderWithKwargs()
    eval.run(
        model,
        output_folder="tests/results",
        overwrite_results=True,
        encode_kwargs=my_encode_kwargs,
    )


@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_run_using_benchmark(model: mteb.Encoder):
    """Test that a benchmark object can be run using the MTEB class."""
    bench = Benchmark(name="test_bench", tasks=["STS12", "SummEval"])

    eval = mteb.MTEB(tasks=bench)
    eval.run(
        model, output_folder="tests/results", overwrite_results=True
    )  # we just want to test that it runs
