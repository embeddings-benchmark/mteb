from __future__ import annotations

import logging

import pytest

import mteb
from mteb import MTEB
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.model_meta import ModelMeta

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task_name", ["BornholmBitextMining"])
@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_reproducibility_workflwo(task_name: str, model_name: str):
    """Test that a model and a task can be fetched and run in a reproducible fashion."""
    model_meta = mteb.get_model(model_name)
    task = mteb.get_task(task_name)

    assert isinstance(model_meta, ModelMeta)
    assert isinstance(task, mteb.AbsTask)

    model = model_meta.load_model()
    assert isinstance(model, (Encoder, EncoderWithQueryCorpusEncode))

    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
