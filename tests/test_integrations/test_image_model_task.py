import logging

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.task_grid import MOCK_MIEB_TASK_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/random-encoder-baseline")])
def test_image_model_task_integration(task: AbsTask, model: mteb.Encoder):
    """Test that image models and image tasks integrate"""
    mteb.evaluate(model, task, cache=None)
