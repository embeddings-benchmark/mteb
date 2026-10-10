"""Similarity is scored on the device the model was loaded on, not the one it reports later.

`start_multi_process_pool` moves the main model to the CPU, so the wrappers capture the device
when they are created.
"""

import warnings

import numpy as np
import pytest
import torch

import mteb.models.sentence_transformer_wrapper as wrapper_module
from mteb.models import (
    ModelMeta,
    MultiVectorWrapper,
    SentenceTransformerEncoderWrapper,
    SparseEncoderWrapper,
)
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

CPU = torch.device("cpu")
LOAD_DEVICE = torch.device("meta")  # stands in for the GPU the model was loaded on


class FakeModel:
    """Minimal stand-in for a Sentence Transformers model that records where it scores."""

    prompts: dict[str, str] = {}
    default_prompt_name = None

    def __init__(self, device: torch.device | str = LOAD_DEVICE) -> None:
        self.device = torch.device(device)
        self.scored_on: list[torch.device] = []

    def similarity(
        self, a: torch.Tensor, b: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        self.scored_on.append(device or a.device)
        return torch.zeros(len(a), len(b))

    def start_pool(self) -> None:
        """`start_multi_process_pool` moves the main model to the CPU."""
        self.device = CPU


class WrapperWithoutParentInit(SentenceTransformerEncoderWrapper):
    """Like subclasses that set up their own model and never call the parent `__init__`."""

    def __init__(self, model: FakeModel) -> None:
        self.model = model


@pytest.fixture(autouse=True)
def fake_model_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrappers build their `ModelMeta` from the real model, which `FakeModel` is not."""
    for name in (
        "from_sentence_transformer_model",
        "from_sparse_encoder_model",
        "from_multi_vector_encoder_model",
    ):
        monkeypatch.setattr(
            ModelMeta, name, classmethod(lambda cls, model: ModelMeta.create_empty())
        )


@pytest.mark.parametrize(
    "wrapper_cls", [SentenceTransformerEncoderWrapper, SparseEncoderWrapper]
)
def test_dense_and_sparse_score_on_device_captured_at_load(wrapper_cls) -> None:
    model = FakeModel()
    wrapper = wrapper_cls(model)
    model.start_pool()

    scores = wrapper.similarity(np.zeros((2, 4), dtype=np.float32), torch.zeros(3, 4))

    assert model.scored_on == [LOAD_DEVICE]
    assert scores.device == CPU
    assert tuple(scores.shape) == (2, 3)


def test_multi_vector_scores_on_device_captured_at_load() -> None:
    model = FakeModel()
    wrapper = MultiVectorWrapper(model)
    model.start_pool()

    wrapper.similarity([torch.zeros(2, 4)], [torch.zeros(3, 4)])

    assert model.scored_on == [LOAD_DEVICE]


def test_instruct_model_scores_on_device_captured_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *a, **k: model
    )
    wrapper = InstructSentenceTransformerModel("model", "revision")
    model.start_pool()

    scores = wrapper.similarity(torch.zeros(2, 4), torch.zeros(3, 4))

    assert model.scored_on == [LOAD_DEVICE]
    assert tuple(scores.shape) == (2, 3)


def test_device_is_captured_on_first_use_when_parent_init_was_skipped() -> None:
    model = FakeModel(CPU)
    wrapper = WrapperWithoutParentInit(model)

    wrapper.similarity(torch.ones(2, 4), torch.ones(3, 4))

    assert model.scored_on == [CPU]


@pytest.mark.parametrize("sparse", [False, True])
def test_corpus_is_scored_in_blocks(
    monkeypatch: pytest.MonkeyPatch, sparse: bool
) -> None:
    monkeypatch.setattr(wrapper_module, "_SCORING_BLOCK_ELEMENTS", 8)  # 2 rows of dim 4
    block_sizes: list[int] = []

    class BlockModel(FakeModel):
        def similarity(self, a, b, device=None):  # type: ignore[override]
            block_sizes.append(b.shape[0])
            return a.to_dense() @ b.to_dense().T

    queries, corpus = torch.randn(3, 4), torch.randn(5, 4)
    expected = queries @ corpus.T
    if sparse:
        queries, corpus = queries.to_sparse(), corpus.to_sparse()
    wrapper = SentenceTransformerEncoderWrapper(BlockModel(CPU))

    scores = wrapper.similarity(queries, corpus)

    assert block_sizes == [2, 2, 1]
    torch.testing.assert_close(scores, expected)


def test_single_embedding_is_scored_as_a_collection_of_one() -> None:
    wrapper = SentenceTransformerEncoderWrapper(FakeModel(CPU))

    scores = wrapper.similarity(
        np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )

    assert tuple(scores.shape) == (1, 1)


@pytest.mark.parametrize(
    "wrapper_cls", [SentenceTransformerEncoderWrapper, SparseEncoderWrapper]
)
def test_sparse_inputs_are_scored_on_cpu_on_mps(wrapper_cls) -> None:
    model = FakeModel("mps")  # sparse tensors can't be moved there
    wrapper = wrapper_cls(model)

    wrapper.similarity(torch.randn(2, 4).to_sparse(), torch.randn(3, 4).to_sparse())

    assert model.scored_on == [CPU]


def test_sparse_encoder_is_scored_on_cpu_on_mps_even_with_dense_inputs() -> None:
    model = FakeModel("mps")
    wrapper = SparseEncoderWrapper(model)

    wrapper.similarity(torch.randn(2, 4), torch.randn(3, 4))

    assert model.scored_on == [CPU]


def test_embeddings_already_on_an_accelerator_are_scored_as_is() -> None:
    model = FakeModel(CPU)
    wrapper = SentenceTransformerEncoderWrapper(model)

    wrapper.similarity(
        torch.zeros(2, 4, device="meta"), torch.zeros(3, 4, device="meta")
    )

    assert model.scored_on == [LOAD_DEVICE]


@pytest.mark.parametrize("accelerator_available", [True, False])
def test_warns_when_model_on_cpu_but_accelerator_available(
    monkeypatch: pytest.MonkeyPatch, accelerator_available: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: accelerator_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper = SentenceTransformerEncoderWrapper(FakeModel(CPU))

    warned = any("scored on the CPU" in str(w.message) for w in caught)
    assert warned == accelerator_available
    assert wrapper._scoring_device == CPU


def test_model_without_device_is_scored_on_cpu() -> None:
    class ModelWithoutDevice:
        prompts: dict[str, str] = {}
        default_prompt_name = None

    wrapper = SentenceTransformerEncoderWrapper(ModelWithoutDevice())

    assert wrapper._scoring_device == CPU
