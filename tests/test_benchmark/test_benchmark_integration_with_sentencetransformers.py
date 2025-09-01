"""test mteb.MTEB's integration with SentenceTransformers"""

from __future__ import annotations

import logging
from pathlib import Path

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
def test_benchmark_sentence_transformer(
    task: str | AbsTask, model_name: str, tmp_path: Path
):
    """Test that a task can be fetched and run"""
    model = SentenceTransformer(model_name)
    # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
    # SentenceTransformerWrapper would set the model's prompts to None because
    # the mock tasks are not in the MTEB task registry. The linked PR changes
    # this behavior and keeps the prompts as configured by the model, so this
    # test now sets the prompts to None explicitly to preserve the legacy
    # behavior and focus the test on the tasks instead of the prompts.
    model.prompts = None
    eval = MTEB(tasks=[task])
    eval.run(model, output_folder=tmp_path.as_posix(), overwrite_results=True)
