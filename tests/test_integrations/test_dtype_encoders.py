"""test that mteb works with various output types of encoders"""

import logging

import pytest
import torch
from packaging.version import Version

import mteb
from mteb.abstasks import AbsTask
from tests.mock_models import MockSentenceTransformersbf16Encoder
from tests.task_grid import MOCK_TASK_TEST_GRID_MONOLINGUAL

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID_MONOLINGUAL)
@pytest.mark.parametrize(
    "model",
    [
        mteb.get_model("mteb/baseline-random-encoder"),
        mteb.get_model(
            "mteb/baseline-random-encoder",
            array_framework="torch",
            dtype=torch.float32,
        ),
        mteb.get_model(
            "mteb/baseline-random-encoder",
            array_framework="torch",
            dtype=torch.float16,
        ),
        MockSentenceTransformersbf16Encoder(),
    ],
)
def test_encoder_dtype_on_task(task: AbsTask, model: mteb.EncoderProtocol):
    if Version(torch.__version__) == Version("2.0.0"):
        pytest.xfail('Torch will raise "clamp_min_scalar_cpu" not implemented for Half')
    mteb.evaluate(model, task, cache=None)
