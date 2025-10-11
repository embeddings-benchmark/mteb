"""test that prompts are correctly called and passed to the encoders"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from torch.utils.data import DataLoader

import mteb
import mteb.overview
from mteb.abstasks import AbsTask

from ..integration_tests.task_grid import MOCK_TASK_TEST_GRID
from .mock_models import (
    MockSentenceTransformer,
    MockSentenceTransformerWrapper,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task_name", MOCK_TASK_TEST_GRID)
def test_prompt_name_passed_to_all_encodes(task_name: str | AbsTask):
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
            assert kwargs["prompt"] is None
            return super().encode(sentences, **kwargs)

    if isinstance(task_name, AbsTask):
        tasks = [task_name]
    else:
        tasks = mteb.get_tasks(tasks=[task_name])

    # Test that the task_name is passed down to the encoder
    model = MockSentenceTransformerWrapper(
        MockEncoderWithInstructions(),
        model_prompts={tasks[0].metadata.name: tasks[0].metadata.name},
    )

    mteb.evaluate(model, tasks, cache=None)

    # Test that the task_name is not passed down to the encoder
    model = EncoderWithoutInstructions()
    assert model.prompts == {}, "The encoder should not have any prompts"
    mteb.evaluate(model, tasks, cache=None)
