"""test mteb.MTEB's integration with SentenceTransformers"""

from __future__ import annotations

import logging

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask

from .mock_models import MockNumpyEncoder
from .task_grid import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", TASK_TEST_GRID)
@pytest.mark.parametrize("model", [MockNumpyEncoder()])
def test_benchmark_sentence_transformer(task: str | AbsTask, model: mteb.Encoder):
    """Test that a task can be fetched and run"""
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
