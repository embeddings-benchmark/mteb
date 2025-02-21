"""test mteb.MTEB's integration with datasets"""

from __future__ import annotations

import logging

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask

from ..test_benchmark.mock_models import MockAudioEncoder
from ..test_benchmark.task_grid import MOCK_MAEB_TASK_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID)
@pytest.mark.parametrize("model", [MockAudioEncoder()])
def test_benchmark_audio_encoder(task: str | AbsTask, model: mteb.Encoder):
    """Test that a task can be fetched and run"""
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
