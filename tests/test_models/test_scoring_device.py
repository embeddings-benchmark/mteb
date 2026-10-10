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

    wrapper = SimpleNamespace(
        model=SimpleNamespace(similarity=similarity),
        _scoring_device=torch.device("cpu"),
    )

    scores = module._score_on_device(wrapper, queries, corpus)

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


def test_single_embedding_is_scored_as_a_collection_of_one() -> None:
    from mteb.models import sentence_transformer_wrapper as module

    wrapper = SimpleNamespace(
        model=SimpleNamespace(similarity=lambda a, b: a @ b.T),
        _scoring_device=torch.device("cpu"),
    )

    scores = module._score_on_device(
        wrapper, torch.ones(4).numpy(), torch.ones(4).numpy()
    )

    assert tuple(scores.shape) == (1, 1)
    assert scores.item() == 4


def test_scoring_device_is_resolved_when_init_was_skipped() -> None:
    """Subclasses that don't call the parent `__init__` never captured the device."""
    wrapper = object.__new__(SentenceTransformerEncoderWrapper)
    wrapper.model = SimpleNamespace(
        device=torch.device("cpu"), similarity=lambda a, b: a @ b.T
    )

    scores = wrapper.similarity(torch.ones(2, 4), torch.ones(3, 4))

    assert tuple(scores.shape) == (2, 3)
    assert wrapper._scoring_device == torch.device("cpu")


def test_model_without_device_is_scored_on_cpu() -> None:
    from mteb.models import sentence_transformer_wrapper as module

    assert module._capture_scoring_device(SimpleNamespace()) == torch.device("cpu")


def test_sparse_inputs_are_scored_on_cpu_on_mps() -> None:
    from mteb.models import sentence_transformer_wrapper as module

    seen: list[torch.device] = []

    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen.append(a.device)
        return a.to_dense() @ b.to_dense().T

    wrapper = SimpleNamespace(
        model=SimpleNamespace(similarity=similarity),
        _scoring_device=torch.device("mps"),  # sparse tensors can't be moved there
    )
    queries, corpus = torch.randn(2, 4).to_sparse(), torch.randn(3, 4).to_sparse()

    scores = module._score_on_device(wrapper, queries, corpus)

    assert seen == [torch.device("cpu")]
    assert tuple(scores.shape) == (2, 3)


def test_sparse_encoder_is_scored_on_cpu_on_mps() -> None:
    from mteb.models import sentence_transformer_wrapper as module

    seen: list[torch.device] = []

    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen.append(a.device)
        return a @ b.T

    wrapper = object.__new__(SparseEncoderWrapper)
    wrapper.model = SimpleNamespace(similarity=similarity)
    wrapper._scoring_device = torch.device("mps")

    module._score_on_device(wrapper, torch.randn(2, 4), torch.randn(3, 4))

    assert seen == [torch.device("cpu")]


def test_embeddings_already_on_an_accelerator_are_scored_as_is() -> None:
    from mteb.models import sentence_transformer_wrapper as module

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append((a, b))
        return torch.zeros(len(a), len(b))

    wrapper = SimpleNamespace(
        model=SimpleNamespace(similarity=similarity),
        _scoring_device=torch.device("cpu"),
    )
    queries, corpus = torch.zeros(2, 4, device="meta"), torch.zeros(3, 4, device="meta")

    module._score_on_device(wrapper, queries, corpus)

    assert calls == [(queries, corpus)]


def test_instruct_model_scores_with_model_similarity_on_captured_device() -> None:
    from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

    seen: list[torch.device] = []
    wrapper = object.__new__(InstructSentenceTransformerModel)
    wrapper.model = _fake_model(seen)
    wrapper._scoring_device = SCORING_DEVICE

    scores = wrapper.similarity(torch.zeros(2, 4), torch.zeros(3, 4))

    assert seen == [SCORING_DEVICE, SCORING_DEVICE]
    assert tuple(scores.shape) == (2, 3)
