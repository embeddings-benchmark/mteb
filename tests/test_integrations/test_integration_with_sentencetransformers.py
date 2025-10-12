"""test that mteb.evaluate integrates with SentenceTransformers"""

import logging

import pytest
from sentence_transformers import SentenceTransformer

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask
from tests.mock_models import MockCLIPEncoder
from tests.task_grid import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize("model_name", ["average_word_embeddings_levy_dependency"])
def test_sentence_transformer_integration(task: AbsTask, model_name: str):
    """Test that a task can be fetched and run"""
    model = SentenceTransformer(model_name)
    # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
    # SentenceTransformerWrapper would set the model's prompts to None because
    # the mock tasks are not in the MTEB task registry. The linked PR changes
    # this behavior and keeps the prompts as configured by the model, so this
    # test now sets the prompts to an empty dict explicitly to preserve the legacy
    # behavior and focus the test on the tasks instead of the prompts.
    # Using empty dict instead of None to avoid TypeError in SentenceTransformers 5.0.0+
    model.prompts = {}
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID)
@pytest.mark.parametrize("model", [MockCLIPEncoder()])
def test_benchmark_sentence_transformer(task: AbsTask, model: mteb.Encoder):
    """Test that a task can be fetched and run"""
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
