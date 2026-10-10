import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mteb.models.sentence_transformer_wrapper import (
    MultiVectorWrapper,
    SentenceTransformerEncoderWrapper,
    SparseEncoderWrapper,
)

SCORING_DEVICE = torch.device("meta")


def _fake_model(seen: list[torch.device]) -> SimpleNamespace:
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen.append(a.device)
        seen.append(b.device)
        return torch.zeros(len(a), len(b))

    # Simulates `start_multi_process_pool`, which moves the main model to the CPU.
    return SimpleNamespace(device=torch.device("cpu"), similarity=similarity)


@pytest.mark.parametrize(
    "wrapper_cls", [SentenceTransformerEncoderWrapper, SparseEncoderWrapper]
)
def test_similarity_runs_on_device_captured_at_load(wrapper_cls) -> None:
    seen: list[torch.device] = []
    wrapper = object.__new__(wrapper_cls)
    wrapper.model = _fake_model(seen)
    wrapper._scoring_device = SCORING_DEVICE  # captured before the pool moved the model

    scores = wrapper.similarity(np.zeros((2, 4), dtype=np.float32), torch.zeros(3, 4))

    assert seen == [SCORING_DEVICE, SCORING_DEVICE]
    assert scores.device.type == "cpu"
    assert tuple(scores.shape) == (2, 3)


def test_multi_vector_uses_captured_device() -> None:
    seen: list[torch.device] = []

    def similarity(queries, documents, device=None):
        seen.append(device)
        return torch.zeros(len(queries), len(documents))

    wrapper = object.__new__(MultiVectorWrapper)
    wrapper.model = SimpleNamespace(device=torch.device("cpu"), similarity=similarity)
    wrapper._scoring_device = torch.device("cpu:0")  # distinguishable from model.device

    wrapper.similarity([torch.zeros(2, 4)], [torch.zeros(3, 4)])

    assert seen == [torch.device("cpu:0")]


@pytest.mark.parametrize("sparse", [False, True])
def test_corpus_is_scored_in_blocks(monkeypatch, sparse: bool) -> None:
    from mteb.models import sentence_transformer_wrapper as module

    monkeypatch.setattr(module, "_SCORING_BLOCK_ELEMENTS", 8)  # 2 rows of dim 4
    block_sizes: list[int] = []

    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        block_sizes.append(b.shape[0])
        return a.to_dense() @ b.to_dense().T if sparse else a @ b.T

    queries = torch.randn(3, 4)
    corpus = torch.randn(5, 4)
    if sparse:
        queries, corpus = queries.to_sparse(), corpus.to_sparse()

    scores = module._score_on_device(similarity, "cpu", queries, corpus)

    assert block_sizes == [2, 2, 1]
    expected = (
        queries.to_dense() @ corpus.to_dense().T if sparse else queries @ corpus.T
    )
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("accelerator_available", [True, False])
def test_warns_when_model_on_cpu_but_accelerator_available(
    monkeypatch, accelerator_available: bool
) -> None:
    from mteb.models import sentence_transformer_wrapper as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: accelerator_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    model = SimpleNamespace(device=torch.device("cpu"))

    if accelerator_available:
        with pytest.warns(UserWarning, match="scored on the CPU"):
            device = module._capture_scoring_device(model)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = module._capture_scoring_device(model)
    assert device == torch.device("cpu")
