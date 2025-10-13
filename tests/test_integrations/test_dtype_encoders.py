"""test that mteb works with various output types of encoders"""

from __future__ import annotations

import logging

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.mock_models import MockSentenceTransformersbf16Encoder
from tests.task_grid import MOCK_TASK_TEST_GRID_MONOLINGUAL

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID_MONOLINGUAL)
@pytest.mark.parametrize(
    "model",
    [
        mteb.get_model("mock/random-encoder-baseline"),
        mteb.get_model("mock/random-torch-encoder-baseline"),
        mteb.get_model("mock/random-torch-fp16-encoder-baseline"),
        MockSentenceTransformersbf16Encoder(),
    ],
)
def test_encoder_dtype_on_task(task: AbsTask, model: mteb.Encoder):
    mteb.evaluate(model, task, cache=None)
