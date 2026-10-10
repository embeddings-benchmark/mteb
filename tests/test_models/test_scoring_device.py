"""Similarity is scored on the device the model was loaded on, not the one it reports later.

`start_multi_process_pool` moves the main model to the CPU, so the wrappers capture the device
when they are created.
"""

import numpy as np
import pytest
import sentence_transformers
import torch
from packaging.version import Version

import mteb.models.sentence_transformer_wrapper as wrapper_module
from mteb.models import (
    MultiVectorWrapper,
    SentenceTransformerEncoderWrapper,
    SparseEncoderWrapper,
)
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.sentence_transformer_wrapper import (
    SENTENCE_TRANSFORMERS_MULTI_VECTOR_VERSION,
    SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION,
)
from tests.mock_models import MockSentenceTransformer

CPU = torch.device("cpu")
LOAD_DEVICE = torch.device("meta")  # stands in for the GPU the model was loaded on


class FakeSentenceTransformer(MockSentenceTransformer):
    """Records where `similarity` is computed."""

    truncate_dim = None  # read by the sparse and multi-vector wrappers

    def __init__(self, device: torch.device | str = LOAD_DEVICE) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.scored_on: list[torch.device] = []

    @property
    def device(self) -> torch.device:
        return self._device

    def start_pool(self) -> None:
        """`start_multi_process_pool` moves the main model to the CPU."""
        self._device = CPU

    def similarity(self, a, b, device=None):
        self.scored_on.append(device or a.device)
        return torch.zeros(len(a), len(b))


def _wrapper_cls(kind: str) -> type:
    """The wrapper for `kind`, skipping if Sentence Transformers is too old for it."""
    if kind == "sparse" and _version_below(SENTENCE_TRANSFORMERS_QUERY_ENCODE_VERSION):
        pytest.skip("SparseEncoderWrapper needs a newer sentence-transformers")
    if kind == "multi_vector" and _version_below(
        SENTENCE_TRANSFORMERS_MULTI_VECTOR_VERSION
    ):
        pytest.skip("MultiVectorWrapper needs a newer sentence-transformers")
    return {
        "dense": SentenceTransformerEncoderWrapper,
        "sparse": SparseEncoderWrapper,
        "multi_vector": MultiVectorWrapper,
    }[kind]


def _version_below(minimum: str) -> bool:
    return Version(sentence_transformers.__version__).release < Version(minimum).release


@pytest.mark.parametrize("kind", ["dense", "sparse", "multi_vector"])
def test_scores_on_device_captured_at_load(kind: str) -> None:
    wrapper_cls = _wrapper_cls(kind)
    model = FakeSentenceTransformer()
    wrapper = wrapper_cls(model)
    model.start_pool()
    # Multi-vector embeddings are one (num_tokens, dim) tensor per input
    queries, documents = (
        ([torch.zeros(2, 4)], [torch.zeros(3, 4)])
        if kind == "multi_vector"
        else (np.zeros((2, 4), dtype=np.float32), torch.zeros(3, 4))
    )

    scores = wrapper.similarity(queries, documents)

    assert model.scored_on == [LOAD_DEVICE]
    assert scores.device == CPU


@pytest.mark.parametrize(
    "wrapper_cls",
    [SentenceTransformerEncoderWrapper, InstructSentenceTransformerModel],
)
def test_device_is_captured_on_first_use_when_parent_init_was_skipped(
    wrapper_cls: type,
) -> None:
    """Subclasses may set up their own model and never call the parent `__init__`."""

    class WrapperWithoutParentInit(wrapper_cls):  # type: ignore[valid-type, misc]
        def __init__(self, model: sentence_transformers.SentenceTransformer) -> None:
            self.model = model

    model = FakeSentenceTransformer(CPU)

    scores = WrapperWithoutParentInit(model).similarity(
        torch.ones(2, 4), torch.ones(3, 4)
    )

    assert model.scored_on == [CPU]
    assert tuple(scores.shape) == (2, 3)


@pytest.mark.parametrize("sparse_inputs", [False, True])
def test_corpus_is_scored_in_blocks(sparse_inputs: bool) -> None:
    class ProductModel(FakeSentenceTransformer):
        def __init__(self, device: torch.device | str) -> None:
            super().__init__(device)
            self.block_sizes: list[int] = []

        def similarity(self, a, b, device=None):
            self.block_sizes.append(b.shape[0])
            return a.to_dense() @ b.to_dense().T

    model = ProductModel(CPU)
    wrapper = SentenceTransformerEncoderWrapper(model)
    queries, corpus = torch.randn(3, 4), torch.randn(5, 4)
    expected = queries @ corpus.T
    if sparse_inputs:
        queries, corpus = queries.to_sparse(), corpus.to_sparse()

    scores = wrapper_module._score_on_device(
        wrapper,
        queries,
        corpus,
        block_elements=8,  # 2 rows of dim 4
    )

    assert model.block_sizes == [2, 2, 1]
    torch.testing.assert_close(scores, expected)


def test_embeddings_already_on_an_accelerator_are_scored_as_is() -> None:
    model = FakeSentenceTransformer(CPU)
    wrapper = SentenceTransformerEncoderWrapper(model)

    wrapper.similarity(
        torch.zeros(2, 4, device="meta"), torch.zeros(3, 4, device="meta")
    )

    assert model.scored_on == [LOAD_DEVICE]
