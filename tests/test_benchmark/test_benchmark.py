"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from torch.utils.data import DataLoader

import mteb
import mteb.overview
from mteb.abstasks import AbsTask, TaskMetadata
from mteb.create_meta import generate_readme
from mteb.evaluation.MTEB import logger
from mteb.types import Array, BatchedInput, PromptType

from .mock_models import (
    AbsMockEncoder,
    MockCLIPEncoder,
    MockMocoEncoder,
    MockNumpyEncoder,
    MockSentenceTransformer,
    MockSentenceTransformersbf16Encoder,
    MockSentenceTransformerWrapper,
    MockTorchEncoder,
    MockTorchfp16Encoder,
)
from .mock_tasks import (
    MockImageClusteringTask,
    MockImageTextPairClassificationTask,
    MockInstructionRetrieval,
    MockMultilingualInstructionRetrieval,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockRerankingTask,
    MockRetrievalTask,
)
from .task_grid import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("tasks", [MOCK_TASK_TEST_GRID])
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_mulitple_mteb_tasks(tasks: list[AbsTask], model: mteb.Encoder, tmp_path: Path):
    """Test that multiple tasks can be run"""
    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder=tmp_path.as_posix(), overwrite_results=True)

    # ensure that we can generate a readme from the output folder
    generate_readme(tmp_path)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize(
    "model",
    [
        MockNumpyEncoder(),
        MockTorchEncoder(),
        MockTorchfp16Encoder(),
        MockSentenceTransformersbf16Encoder(),
    ],
)
def test_benchmark_encoders_on_task(
    task: str | AbsTask, model: mteb.Encoder, tmp_path: Path
):
    """Test that a task can be fetched and run using a variety of encoders"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder=tmp_path.as_posix())


@pytest.mark.parametrize("task", [MockMultilingualRetrievalTask()])
@pytest.mark.parametrize(
    "model",
    [MockSentenceTransformer()],
)
def test_run_eval_without_co2_tracking(
    task: str | AbsTask, model: mteb.Encoder, tmp_path: Path
):
    """Test that a task can be fetched and run without CO2 tracking"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    eval.run(model, output_folder=tmp_path.as_posix(), co2_tracker=False)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID[:1])
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_reload_results(task: str | AbsTask, model: mteb.Encoder, tmp_path: Path):
    """Test that when rerunning the results are reloaded correctly"""
    if isinstance(task, str):
        tasks = mteb.get_tasks(tasks=[task])
    else:
        tasks = [task]

    eval = mteb.MTEB(tasks=tasks)
    results = eval.run(model, output_folder=tmp_path.as_posix(), overwrite_results=True)

    assert isinstance(results, list)
    assert isinstance(results[0], mteb.TaskResult)

    # reload the results
    results = eval.run(
        model, output_folder=tmp_path.as_posix(), overwrite_results=False
    )

    assert isinstance(results, list)
    assert isinstance(results[0], mteb.TaskResult)


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_prompt_name_passed_to_all_encodes(task_name: str | AbsTask, tmp_path: Path):
    """Test that all tasks correctly pass down the prompt_name to the encoder which supports it, and that the encoder which does not support it does not
    receive it.
    """
    _task_name = (
        task_name.metadata.name if isinstance(task_name, AbsTask) else task_name
    )

    class MockEncoderWithInstructions(MockSentenceTransformer):
        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            assert prompt_name == _task_name
            return np.zeros((len(sentences.dataset), 10))

    class EncoderWithoutInstructions(MockSentenceTransformer):
        prompts = {}

        def encode(self, sentences: DataLoader, **kwargs):
            assert kwargs["prompt_name"] is None
            return super().encode(sentences, **kwargs)

    if isinstance(task_name, AbsTask):
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
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
    )
    # Test that the task_name is not passed down to the encoder
    model = EncoderWithoutInstructions()
    assert model.prompts == {}, "The encoder should not have any prompts"
    eval.run(model, output_folder=tmp_path.as_posix(), overwrite_results=True)


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_encode_kwargs_passed_to_all_encodes(task_name: str | AbsTask, tmp_path: Path):
    """Test that all tasks correctly pass down the encode_kwargs to the encoder."""
    my_encode_kwargs = {"no_one_uses_this_args": "but_its_here"}

    class MockEncoderWithKwargs(AbsMockEncoder):
        def encode(self, sentences: DataLoader, task_name: str | None = None, **kwargs):
            assert "no_one_uses_this_args" in kwargs
            assert (
                my_encode_kwargs["no_one_uses_this_args"]
                == kwargs["no_one_uses_this_args"]
            )
            return np.zeros((len(sentences.dataset), 10))

    if isinstance(task_name, AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockEncoderWithKwargs()
    eval.run(
        model,
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
        encode_kwargs=my_encode_kwargs,
    )


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID + MOCK_MIEB_TASK_GRID)
def test_task_metadata_passed_encoder(task_name: mteb.AbsTask, tmp_path: Path):
    """Test that all tasks correctly pass down the task_name to the encoder."""
    _task_name = (
        task_name.metadata.name if isinstance(task_name, mteb.AbsTask) else task_name
    )

    class MockEncoder(MockCLIPEncoder):
        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            assert task_metadata.name == _task_name
            assert isinstance(hf_split, str)
            assert isinstance(hf_subset, str)
            return np.zeros((len(inputs.dataset), 10))

    if isinstance(task_name, mteb.AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    eval = mteb.MTEB(tasks=tasks)

    eval.run(
        MockEncoder(),
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
    )


@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_run_using_benchmark(model: mteb.Encoder, tmp_path: Path):
    """Test that a benchmark object can be run using the MTEB class."""
    bench = mteb.Benchmark(
        name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"])
    )

    eval = mteb.MTEB(tasks=[bench])
    eval.run(
        model, output_folder=tmp_path.as_posix(), overwrite_results=True
    )  # we just want to test that it runs


@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_run_using_list_of_benchmark(model: mteb.Encoder, tmp_path: Path):
    """Test that a list of benchmark objects can be run using the MTEB class."""
    bench = [
        mteb.Benchmark(
            name="test_bench", tasks=mteb.get_tasks(tasks=["STS12", "SummEval"])
        )
    ]

    eval = mteb.MTEB(tasks=bench)
    eval.run(
        model, output_folder=tmp_path.as_posix(), overwrite_results=True
    )  # we just want to test that it runs


def test_benchmark_names_must_be_unique():
    import mteb.benchmarks.benchmarks as benchmark_module

    names = [
        inst.name
        for nam, inst in benchmark_module.__dict__.items()
        if isinstance(inst, mteb.Benchmark)
    ]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "name", ["MTEB(eng, v1)", "MTEB(rus, v1)", "MTEB(Scandinavian, v1)"]
)
def test_get_benchmark(name):
    benchmark = mteb.get_benchmark(benchmark_name=name)
    assert isinstance(benchmark, mteb.Benchmark)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize("is_task_name", [True, False])
def test_prompt_name_passed_to_all_encodes_with_prompts(
    task: AbsTask | str, is_task_name: bool, tmp_path: Path
):
    """Test that all tasks and task_types correctly pass down the prompt_name to the encoder with prompts."""
    _task_name = task.metadata.name if isinstance(task, AbsTask) else task

    if isinstance(task, AbsTask):
        tasks = [task]
        _task_type = task.metadata.type
    else:
        tasks = mteb.get_tasks(tasks=[task])
        _task_type = tasks[0].metadata.type

    to_compare = _task_name if is_task_name else _task_type

    class MockEncoderWithPrompts(MockSentenceTransformer):
        prompts = {}

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            assert prompt_name == to_compare
            return np.zeros((len(sentences.dataset), 10))

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts={to_compare: to_compare}
    )
    eval.run(
        model,
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
    )

    class MockEncoderWithExistingPrompts(MockSentenceTransformer):
        prompts = {to_compare: to_compare}

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            assert prompt_name == to_compare
            return np.zeros((len(sentences.dataset), 10))

    eval = mteb.MTEB(tasks=tasks)

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(MockEncoderWithExistingPrompts())
    eval.run(
        model,
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
    )


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockMultilingualRerankingTask(),
        MockInstructionRetrieval(),
        MockMultilingualInstructionRetrieval(),
        MockRetrievalTask(),
        MockMultilingualRetrievalTask(),
    ],
)
@pytest.mark.parametrize("is_task_name", [True, False])
def test_model_query_passage_prompts_task_type(
    task: AbsTask | str, is_task_name: bool, tmp_path: Path
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

    class MockEncoderWithPrompts:
        is_query = True

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            check_prompt(prompt_name, self.is_query)
            self.is_query = not self.is_query
            return np.zeros((len(sentences.dataset), 10))

    class MockSentenceEncoderWithPrompts:
        is_query = True

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            check_prompt(prompt_name, self.is_query)
            self.is_query = not self.is_query
            return np.zeros((len(sentences.dataset), 10))

    eval = mteb.MTEB(tasks=tasks)
    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts=prompt_list
    )

    eval.run(
        model,
        model_prompts=prompt_list,
        output_folder=tmp_path.as_posix(),
    )
    model = MockSentenceTransformerWrapper(
        MockSentenceEncoderWithPrompts(), model_prompts=prompt_list
    )

    eval.run(
        model,
        model_prompts=prompt_list,
        output_folder=tmp_path.as_posix(),
        overwrite_results=True,
    )


# NOTE: Covers image and image-text tasks. Can be extended to cover new mixed-modality task types.
@pytest.mark.parametrize(
    "task", [MockImageTextPairClassificationTask(), MockRetrievalTask()]
)
@patch.object(logger, "info")
def test_task_modality_filtering(mock_logger, task):
    eval = mteb.MTEB(tasks=[task])

    # Run the evaluation
    eval.run(
        model=MockMocoEncoder(),
        output_folder="tests/results",
        overwrite_results=True,
    )

    # Check that the task was skipped and the correct log message was generated
    task_modalities = ", ".join(
        f"'{modality}'" for modality in sorted(task.metadata.modalities)
    )
    mock_logger.assert_called_with(
        f"mock/MockMocoModel only supports ['image'], but the task modalities are [{task_modalities}]."
    )


@pytest.mark.parametrize("task", [MockImageClusteringTask()])
def test_task_modality_filtering_model_modalities_more_than_task_modalities(task):
    eval = mteb.MTEB(tasks=[task])

    # Run the evaluation
    eval.run(
        model=MockCLIPEncoder(),
        output_folder="tests/results",
        overwrite_results=True,
    )
