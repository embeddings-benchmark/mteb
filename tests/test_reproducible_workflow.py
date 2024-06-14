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
@pytest.mark.parametrize("model_revision", ["8b3219a92973c328a8e22fadcfa821b5dc75636a"])
def test_reproducibility_workflow(task_name: str, model_name: str, model_revision: str):
    """Test that a model and a task can be fetched and run in a reproducible fashion."""
    model_meta = mteb.get_model_meta(model_name, revision=model_revision)
    task = mteb.get_task(task_name)

    assert isinstance(model_meta, ModelMeta)
    assert isinstance(task, mteb.AbsTask)

    model = mteb.get_model(model_name, revision=model_revision)
    assert isinstance(model, (Encoder, EncoderWithQueryCorpusEncode))

    eval = MTEB(tasks=[task])
    eval.run(model, output_folder="tests/results", overwrite_results=True)
