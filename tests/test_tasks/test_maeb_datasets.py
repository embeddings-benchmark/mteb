"""test mteb.MTEB's integration with datasets"""

import logging

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.task_grid import MOCK_MAEB_TASK_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID)
@pytest.mark.parametrize("model", [mteb.get_model("baseline/random-encoder-baseline")])
def test_benchmark_audio_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)
