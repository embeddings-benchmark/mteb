"""test that mteb works with various output types of encoders"""

from __future__ import annotations

import logging

import pytest

import mteb
import mteb.overview
from mteb.abstasks import AbsTask
from tests.mock_models import (
    MockNumpyEncoder,
    MockSentenceTransformersbf16Encoder,
    MockTorchEncoder,
    MockTorchfp16Encoder,
)
from tests.task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


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
def test_benchmark_encoders_on_task(task: AbsTask, model: mteb.Encoder):
    mteb.evaluate(model, task, cache=None)
