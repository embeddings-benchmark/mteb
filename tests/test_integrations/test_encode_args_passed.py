"""test that arguments, encode_kwargs are correctly called and passed to the encoders"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from torch.utils.data import DataLoader

import mteb
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from tests.task_grid import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID

if TYPE_CHECKING:
    from mteb.types import Array, BatchedInput

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
def test_encode_kwargs_passed_to_all_encodes(task: AbsTask, tmp_path: Path):
    """Test that all tasks correctly pass down the encode_kwargs to the encoder."""
    my_encode_kwargs = {"no_one_uses_this_args": "but_its_here"}

    class MockEncoderWithKwargs(AbsEncoder):
        def encode(self, sentences: DataLoader, task_name: str | None = None, **kwargs):
            assert "no_one_uses_this_args" in kwargs
            assert (
                my_encode_kwargs["no_one_uses_this_args"]
                == kwargs["no_one_uses_this_args"]
            )
            return np.zeros((len(sentences.dataset), 10))

    # Test that the task_name is passed down to the encoder
    model = MockEncoderWithKwargs()
    mteb.evaluate(
        model,
        task,
        encode_kwargs=my_encode_kwargs,
        cache=None,
    )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID + MOCK_MIEB_TASK_GRID)
def test_task_metadata_passed_encoder(task: mteb.AbsTask, tmp_path: Path):
    """Test that all tasks correctly pass down the task_name to the encoder."""
    _task_name = task.metadata.name
    if _task_name in [t.metadata.name for t in MOCK_MIEB_TASK_GRID]:
        pytest.importorskip("PIL")

    class MockEncoder(AbsEncoder):
        mteb_model_meta = ModelMeta(
            loader=None,
            name="no_model_name/available",
            revision="no_revision_available",
            reference=None,
            release_date=None,
            languages=None,
            license=None,
            framework=[],
            training_datasets=None,
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            modalities=[],
        )

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

    mteb.evaluate(
        MockEncoder(),
        task,
        cache=None,
    )
