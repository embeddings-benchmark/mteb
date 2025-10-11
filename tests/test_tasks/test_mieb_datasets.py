"""test mteb.MIEB's integration with datasets"""

<<<<<<< HEAD
# TODO: KCE: Is this test needed? I would probably delete
from __future__ import annotations

=======
>>>>>>> 5b4c6da2eee9d799693184bdfb95447187187f5c
import logging

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask

from ..test_integrations.mock_models import MockCLIPEncoder
from ..integration_tests.task_grid import MOCK_MIEB_TASK_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID)
@pytest.mark.parametrize("model", [MockCLIPEncoder()])
def test_benchmark_sentence_transformer(task: str | AbsTask, model: mteb.Encoder):
    """Test that a task can be fetched and run"""
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
