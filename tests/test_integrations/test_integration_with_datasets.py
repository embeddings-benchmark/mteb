"""test that mteb.evaluate's integration with datasets"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.task_grid import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", TASK_TEST_GRID)
@pytest.mark.parametrize("model", [mteb.get_model("baseline/random-encoder-baseline")])
def test_benchmark_datasets(task: AbsTask, model: mteb.EncoderProtocol, tmp_path: Path):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)
