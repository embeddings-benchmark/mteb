"""test that arguments, encode_kwargs are correctly called and passed to the encoders"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from torch.utils.data import DataLoader

import mteb
import mteb.overview
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import Array, BatchedInput, PromptType
from tests.mock_models import AbsMockEncoder, MockCLIPEncoder
from tests.task_grid import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


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
