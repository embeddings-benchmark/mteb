"""test that prompts, and query/corpus indicators are correctly called and passed to the encoders"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

import mteb
from mteb.abstasks import AbsTask
from mteb.models.abs_encoder import AbsEncoder
from tests.mock_models import (
    MockSentenceTransformer,
    MockSentenceTransformerWrapper,
)
from tests.mock_tasks import (
    MockInstructionRetrieval,
    MockMultilingualInstructionRetrieval,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockRerankingTask,
    MockRetrievalTask,
)
from tests.task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize("is_task_name", [True, False])
def test_prompt_name_passed_to_all_encodes_with_prompts(
    task: AbsTask | str, is_task_name: bool, tmp_path: Path
):
    """Test that all tasks and task_types correctly pass down the prompt_name to the encoder with prompts."""
    _task_name = task.metadata.name if isinstance(task, AbsTask) else task

    tasks = [task]
    _task_type = task.metadata.type

    to_compare = _task_name if is_task_name else _task_type

    class MockEncoderWithPrompts(MockSentenceTransformer):
        prompts = {}

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            assert prompt_name == to_compare
            return np.zeros((len(sentences.dataset), 10))

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts={to_compare: to_compare}
    )
    mteb.evaluate(
        model,
        tasks,
        cache=None,
    )

    class MockEncoderWithExistingPrompts(MockSentenceTransformer):
        prompts = {to_compare: to_compare}

        def encode(
            self, sentences: DataLoader, prompt_name: str | None = None, **kwargs
        ):
            assert prompt_name == to_compare
            return np.zeros((len(sentences.dataset), 10))

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(MockEncoderWithExistingPrompts())
    mteb.evaluate(
        model,
        tasks,
        cache=None,
    )


@pytest.mark.parametrize("task_name", ["NQ-NL-query", "NQ-NL-document"])
def test_prompt_name_split_correctly(task_name: str, tmp_path: Path):
    """Test that the task name is split correctly into task name and prompt type
    for tasks with multiple `-` in their names.
    """
    AbsEncoder.validate_task_to_prompt_name({task_name: task_name})


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
        prompt_type = "query" if is_query else "document"
        assert prompt_name == f"{task_name}-{prompt_type}"

    prompt_list = {
        f"{task_name}-query": "query",
        f"{task_name}-document": "document",
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

    model = MockSentenceTransformerWrapper(
        MockEncoderWithPrompts(), model_prompts=prompt_list
    )

    mteb.evaluate(
        model,
        tasks,
        cache=None,
    )
    model = MockSentenceTransformerWrapper(
        MockSentenceEncoderWithPrompts(), model_prompts=prompt_list
    )

    mteb.evaluate(
        model,
        tasks,
        cache=None,
    )
