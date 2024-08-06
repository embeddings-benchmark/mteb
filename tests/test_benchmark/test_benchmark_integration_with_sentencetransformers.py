"""test mteb.MTEB's integration with SentenceTransformers"""

from __future__ import annotations

import logging

import pytest
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask

from .task_grid import MOCK_TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
@pytest.mark.parametrize(
    "model_name",
    [
        "average_word_embeddings_levy_dependency",
    ],
)
def test_benchmark_sentence_transformer(task: str | AbsTask, model_name: str):
    """Test that a task can be fetched and run"""
    if isinstance(model_name, str):
        model = SentenceTransformer(model_name)
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
