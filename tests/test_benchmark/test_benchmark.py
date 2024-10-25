"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

import mteb
import mteb.overview
from mteb.benchmarks.benchmarks import Benchmark
from mteb.create_meta import generate_readme

from .mock_models import (
    MockNumpyEncoder,
    MockSentenceTransformer,
    MockSentenceTransformerWrapper,
    MockTorchbf16Encoder,
    MockTorchEncoder,
)
from .mock_tasks import (
    MockInstructionRetrival,
    MockMultilingualInstructionRetrival,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockRerankingTask,
    MockRetrievalTask,
)
from .task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("tasks", [MOCK_TASK_TEST_GRID])
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_mulitple_mteb_tasks(
    tasks: list[mteb.AbsTask], model: mteb.Encoder, tmp_path: Path
):
    """Test that multiple tasks can be run"""
    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder=str(tmp_path), overwrite_results=True)

    # ensure that we can generate a readme from the output folder
    generate_readme(tmp_path)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize(
    "model",
    [
        MockNumpyEncoder(),
        MockTorchEncoder(),
        MockTorchbf16Encoder(),
    ],
)
def test_benchmark_encoders_on_task(task: str | mteb.AbsTask, model: mteb.Encoder):
    """Test that a task can be fetched and run using a variety of encoders"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder="tests/results", overwrite_results=True)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID[:1])
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_reload_results(task: str | mteb.AbsTask, model: mteb.Encoder, tmp_path: Path):
    """Test that when rerunning the results are reloaded correctly"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    results = eval.run(model, output_folder=str(tmp_path), overwrite_results=True)

    assert isinstance(results, list)
    assert isinstance(results[0], mteb.TaskResult)

    # reload the results
    results = eval.run(model, output_folder=str(tmp_path), overwrite_results=False)

    assert isinstance(results, list)
    assert isinstance(results[0], mteb.TaskResult)


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_prompt_name_passed_to_all_encodes(task_name: str | mteb.AbsTask):
    """Test that all tasks correctly pass down the prompt_name to the encoder which supports it, and that the encoder which does not support it does not
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
            assert kwargs["prompt_name"] is None
            return super().encode(sentences, **kwargs)

    if isinstance(task_name, mteb.AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(
        MockEncoderWithInstructions(),
        model_prompts={tasks[0].metadata.name: tasks[0].metadata.name},
    )

    eval.run(
        model,
        output_folder="tests/results",
        overwrite_results=True,
    )
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
            assert "no_one_uses_this_args" in kwargs
            assert (
                my_encode_kwargs["no_one_uses_this_args"]
                == kwargs["no_one_uses_this_args"]
            )
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
    bench = Benchmark(
        name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"])
    )

    eval = mteb.MTEB(tasks=bench)
    eval.run(
        model, output_folder="tests/results", overwrite_results=True
    )  # we just want to test that it runs


@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_run_using_list_of_benchmark(model: mteb.Encoder):
    """Test that a list of benchmark objects can be run using the MTEB class."""
    bench = [
        Benchmark(name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"]))
    ]

    eval = mteb.MTEB(tasks=bench)
    eval.run(
        model, output_folder="tests/results", overwrite_results=True
    )  # we just want to test that it runs


def test_benchmark_names_must_be_unique():
    import mteb.benchmarks.benchmarks as benchmark_module

    names = [
        inst.name
        for nam, inst in benchmark_module.__dict__.items()
        if isinstance(inst, Benchmark)
    ]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", ["MTEB(eng)", "MTEB(rus)", "MTEB(Scandinavian)"])
def test_get_benchmark(name):
    benchmark = mteb.get_benchmark(benchmark_name=name)
    assert isinstance(benchmark, mteb.Benchmark)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize("is_task_name", [True, False])
def test_prompt_name_passed_to_all_encodes_with_prompts(
    task: mteb.AbsTask | str, is_task_name: bool
):
    """Test that all tasks and task_types correctly pass down the prompt_name to the encoder with prompts."""
    _task_name = task.metadata.name if isinstance(task, mteb.AbsTask) else task

    if isinstance(task, mteb.AbsTask):
        tasks = [task]
        _task_type = task.metadata.type
    else:
        tasks = mteb.get_tasks(tasks=[task])
        _task_type = tasks[0].metadata.type

    to_compare = _task_name if is_task_name else _task_type

    class MockEncoderWithPrompts(mteb.Encoder):
        prompts = {}

        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert prompt_name == to_compare
            return np.zeros((len(sentences), 10))

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts={to_compare: to_compare}
    )
    eval.run(
        model,
        output_folder="tests/results",
        overwrite_results=True,
    )

    class MockEncoderWithExistingPrompts(mteb.Encoder):
        prompts = {to_compare: to_compare}

        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            assert prompt_name == to_compare
            return np.zeros((len(sentences), 10))

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(MockEncoderWithExistingPrompts())
    eval.run(
        model,
        output_folder="tests/results",
        overwrite_results=True,
    )


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockMultilingualRerankingTask(),
        MockInstructionRetrival(),
        MockMultilingualInstructionRetrival(),
        MockRetrievalTask(),
        MockMultilingualRetrievalTask(),
    ],
)
@pytest.mark.parametrize("is_task_name", [True, False])
def test_model_query_passage_prompts_task_type(
    task: mteb.AbsTask | str, is_task_name: bool
):
    """Test that the model with prompts is correctly called."""
    tasks = [task]

    task_name = task.metadata.name if is_task_name else task.metadata.type

    def check_prompt(prompt_name, is_query):
        prompt_type = "query" if is_query else "passage"
        assert prompt_name == f"{task_name}-{prompt_type}"

    prompt_list = {
        f"{task_name}-query": "query",
        f"{task_name}-passage": "passage",
    }

    class MockEncoderWithPrompts(mteb.Encoder):
        is_query = True

        def encode(self, sentences, prompt_name: str | None = None, **kwargs):
            check_prompt(prompt_name, self.is_query)
            self.is_query = not self.is_query
            return np.zeros((len(sentences), 10))

    class MockSentenceEncoderWithPrompts(MockSentenceTransformer):
        is_query = True

        def encode(self, sentences, prompt_name: str | None = None, *args, **kwargs):
            check_prompt(prompt_name, self.is_query)
            self.is_query = not self.is_query
            return torch.randn(len(sentences), 10).numpy()

    eval = mteb.MTEB(tasks=tasks)
    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts=prompt_list
    )

    eval.run(
        model,
        model_prompts=prompt_list,
        output_folder="tests/results",
        overwrite_results=True,
    )
    model = MockSentenceTransformerWrapper(
        MockSentenceEncoderWithPrompts(), model_prompts=prompt_list
    )

    eval.run(
        model,
        model_prompts=prompt_list,
        output_folder="tests/results",
        overwrite_results=True,
    )
