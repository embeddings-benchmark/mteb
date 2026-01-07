from pathlib import Path

import pytest

import mteb
from mteb import AbsTask
from mteb.models.vllm_wrapper import VllmCrossEncoderWrapper, VllmEncoderWrapper
from tests.mock_tasks import MockRerankingTask


@pytest.mark.parametrize("model_name", ["cross-encoder/ms-marco-TinyBERT-L2-v2"])
@pytest.mark.parametrize("task", [MockRerankingTask()])
def test_vllm_cross_encoder(task: AbsTask, model_name: str, tmp_path: Path):
    pytest.importorskip("vllm", reason="vllm not installed")

    model = VllmCrossEncoderWrapper(model_name)
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
@pytest.mark.parametrize("task", [MockRerankingTask()])
def test_vllm_encoder(task: AbsTask, model_name: str, tmp_path: Path):
    pytest.importorskip("vllm", reason="vllm not installed")

    model = VllmEncoderWrapper(model_name)
    mteb.evaluate(model, task, cache=None)
