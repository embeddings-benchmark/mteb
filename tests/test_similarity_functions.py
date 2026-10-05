"""Tests for the similarity primitives in ``mteb.similarity_functions``.

Relevance scores must be computed in float32 even when the encoder returns
low-precision (float16/bfloat16) embeddings, so that reduced precision does not
collapse distinct scores into spurious ties.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
import torch
from packaging.version import Version

from mteb.similarity_functions import (
    _max_sim_ragged_core,
    _token_budget_chunks,
    cos_sim,
    dot_score,
    euclidean_sim,
    max_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_max_sim,
)

LOW_PRECISION_DTYPES = [torch.float16, torch.bfloat16]


@pytest.fixture
def embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn(4, 32)
    b = torch.randn(50, 32)
    return a, b


@pytest.mark.parametrize("fn", [cos_sim, dot_score])
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_scores_are_float32_for_low_precision_inputs(fn, dtype, embeddings):
    """Low-precision embeddings must still yield float32 scores (HPS)."""
    a, b = embeddings
    scores = fn(a.to(dtype), b.to(dtype))
    assert scores.dtype == torch.float32


@pytest.mark.parametrize("fn", [pairwise_cos_sim, pairwise_dot_score])
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_pairwise_scores_are_float32_for_low_precision_inputs(fn, dtype):
    torch.manual_seed(0)
    a = torch.randn(16, 32).to(dtype)
    b = torch.randn(16, 32).to(dtype)
    scores = torch.as_tensor(fn(a, b))
    assert scores.dtype == torch.float32


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_hps_collapses_spurious_ties(dtype, embeddings):
    """Upcasting the scoring step recovers the tie structure of float32.

    A naive low-precision matmul buckets scores coarsely and produces spurious
    ties; HPS (upcast-then-score) should recover essentially all of the unique
    scores that full float32 scoring produces.
    """
    if Version(torch.__version__) <= Version("2.5.0"):
        pytest.xfail('Torch will raise "clamp_min_scalar_cpu" not implemented for Half')

    a, b = embeddings
    reference = cos_sim(a, b)  # float32 reference
    hps = cos_sim(a.to(dtype), b.to(dtype))  # low-precision inputs, HPS scoring

    # Naive low-precision scoring (no upcast) for comparison.
    a_norm = torch.nn.functional.normalize(a.to(dtype), p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b.to(dtype), p=2, dim=1)
    naive = a_norm @ b_norm.transpose(0, 1)

    n_ref = torch.unique(reference).numel()
    n_hps = torch.unique(hps).numel()
    n_naive = torch.unique(naive).numel()

    assert n_naive < n_ref, "expected the naive low-precision matmul to create ties"
    assert n_hps >= n_naive, "HPS must not introduce more ties than naive scoring"
    # HPS recovers the vast majority of the distinct float32 scores.
    assert n_hps >= n_ref - 1


@pytest.mark.parametrize("fn", [cos_sim, dot_score])
def test_float32_inputs_are_unchanged(fn, embeddings):
    """HPS is a no-op for embeddings that are already float32."""
    a, b = embeddings
    assert fn(a, b).dtype == torch.float32


@pytest.mark.parametrize("fn", [cos_sim, dot_score])
def test_numpy_inputs_still_supported(fn, embeddings):
    a, b = embeddings
    scores = fn(a.numpy().astype(np.float32), b.numpy().astype(np.float32))
    assert torch.as_tensor(scores).dtype == torch.float32


def _ragged(
    widths: list[int], dim: int = 16, *, non_negative: bool = False
) -> list[torch.Tensor]:
    vectors = [
        torch.nn.functional.normalize(torch.randn(w, dim), dim=-1) for w in widths
    ]
    return [v.abs() for v in vectors] if non_negative else vectors


def test_max_sim_ragged_matches_padded():
    torch.manual_seed(0)
    # Zero padding clips negative token similarities to 0, so the reference only matches
    # padded scores when no similarity is negative.
    queries = _ragged([3, 5, 2, 40], non_negative=True)
    documents = _ragged([7, 4, 9, 6, 5], non_negative=True)
    padded = max_sim(
        torch.nn.utils.rnn.pad_sequence(queries, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(documents, batch_first=True),
    )
    expected = torch.stack(
        [
            torch.stack([(q @ d.T).max(-1).values.sum() for d in documents])
            for q in queries
        ]
    )
    torch.testing.assert_close(max_sim(queries, documents), padded)
    torch.testing.assert_close(padded, expected)


def test_max_sim_ragged_grouped_matches_padded():
    """Scoring in length groups agrees with padding, also when split over several groups."""
    torch.manual_seed(0)
    queries = _ragged([2, 3, 2, 60, 3, 2])
    documents = _ragged([5, 8, 4, 30, 6])
    padded = max_sim(
        torch.nn.utils.rnn.pad_sequence(queries, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(documents, batch_first=True),
    )
    for max_chunk_tokens in (64, 10_000):
        scores = _max_sim_ragged_core(
            queries, documents, torch.device("cpu"), max_chunk_tokens=max_chunk_tokens
        )
        torch.testing.assert_close(scores, padded)


def test_token_budget_chunks_respect_budget():
    widths = [2, 3, 2, 60, 3, 2, 9]
    chunks = _token_budget_chunks(widths, max_tokens=20)

    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(widths)))
    for chunk in chunks:
        padded_tokens = len(chunk) * max(widths[i] for i in chunk)
        assert len(chunk) == 1 or padded_tokens <= 20
    # the 60 token input is over budget alone, and doesn't pad any other group
    assert [3] in chunks


def test_max_sim_ragged_accepts_tensor_and_list():
    torch.manual_seed(0)
    queries = torch.randn(3, 5, 16)
    documents = _ragged([7, 4, 9])
    scores = max_sim(queries, documents)
    expected = max_sim(list(queries), documents)
    assert scores.shape == (3, 3)
    torch.testing.assert_close(scores, expected)


@pytest.fixture
def st_util() -> ModuleType:
    """`sentence_transformers.util`, skipping when sentence-transformers is missing or too old."""
    from sentence_transformers import util

    if not hasattr(util, "maxsim"):
        pytest.skip("sentence-transformers has no maxsim (requires >=6.0.0)")
    return util


MAX_SIM_WIDTHS = [
    pytest.param([3, 5, 2, 7], [6, 4, 9, 5, 8], id="mixed"),
    pytest.param([4, 4, 4], [6, 6], id="uniform"),
    pytest.param([2, 3, 2, 60, 3], [5, 8, 4, 30, 6, 5], id="skewed"),
]


@pytest.mark.parametrize(("query_widths", "document_widths"), MAX_SIM_WIDTHS)
def test_max_sim_matches_sentence_transformers(
    st_util: ModuleType, query_widths: list[int], document_widths: list[int]
):
    torch.manual_seed(0)
    # Zero padding clips negative token similarities to 0, while sentence-transformers masks
    # padding, so the two only agree when no similarity is negative.
    queries = _ragged(query_widths, non_negative=True)
    documents = _ragged(document_widths, non_negative=True)

    expected = st_util.maxsim(queries, documents)
    torch.testing.assert_close(max_sim(queries, documents), expected)
    torch.testing.assert_close(
        max_sim(
            torch.nn.utils.rnn.pad_sequence(queries, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(documents, batch_first=True),
        ),
        expected,
    )


def test_max_sim_grouped_matches_sentence_transformers(st_util: ModuleType):
    """With a tiny budget the length-grouped path splits queries and documents into groups."""
    torch.manual_seed(0)
    queries = _ragged([2, 3, 2, 60, 3, 2], non_negative=True)
    documents = _ragged([5, 8, 4, 30, 6], non_negative=True)

    torch.testing.assert_close(
        _max_sim_ragged_core(
            queries, documents, torch.device("cpu"), max_chunk_tokens=64
        ),
        st_util.maxsim(queries, documents),
    )


def test_pairwise_max_sim_matches_sentence_transformers(st_util: ModuleType):
    torch.manual_seed(0)
    queries = _ragged([3, 5, 2, 7], non_negative=True)
    documents = _ragged([6, 4, 9, 5], non_negative=True)

    torch.testing.assert_close(
        pairwise_max_sim(queries, documents),
        st_util.maxsim_pairwise(queries, documents),
    )


# Small hand-checkable inputs: a = [[1, 0], [0, 2]], b = [[3, 0], [0, 4]]
A = [[1.0, 0.0], [0.0, 2.0]]
B = [[3.0, 0.0], [0.0, 4.0]]


def test_cos_sim_values():
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(cos_sim(A, B), expected)
    torch.testing.assert_close(pairwise_cos_sim(A, B), torch.ones(2))


def test_dot_score_values():
    torch.testing.assert_close(dot_score(A, B), torch.tensor([[3.0, 0.0], [0.0, 8.0]]))
    torch.testing.assert_close(pairwise_dot_score(A, B), torch.tensor([3.0, 8.0]))


def test_euclidean_sim_values():
    # negative distances, e.g. |(1, 0) - (0, 4)| = sqrt(17) and |(0, 2) - (3, 0)| = sqrt(13)
    expected = -torch.tensor([[2.0, 17.0**0.5], [13.0**0.5, 2.0]])
    torch.testing.assert_close(euclidean_sim(A, B), expected)
    torch.testing.assert_close(pairwise_euclidean_sim(A, B), torch.tensor([-2.0, -2.0]))


def test_max_sim_values():
    # query tokens (1, 0) and (0, 1); document tokens (2, 0), (0, 3), (-1, -1)
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    document = torch.tensor([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
    # each query token takes its best document token: 2 + 3
    torch.testing.assert_close(max_sim(query, document), torch.tensor([[5.0]]))
    torch.testing.assert_close(
        pairwise_max_sim([query], [document]), torch.tensor([5.0])
    )


@pytest.mark.parametrize(
    ("ours", "st_similarity_fn"),
    [
        (cos_sim, lambda util: util.cos_sim),
        (dot_score, lambda util: util.dot_score),
        (euclidean_sim, lambda util: util.euclidean_sim),
    ],
)
def test_similarity_matches_sentence_transformers(
    st_util: ModuleType,
    ours,
    st_similarity_fn,
    embeddings: tuple[torch.Tensor, torch.Tensor],
):
    a, b = embeddings
    expected = st_similarity_fn(st_util)

    torch.testing.assert_close(torch.as_tensor(ours(a, b)), expected(a, b))
    # numpy inputs go through the same conversion
    torch.testing.assert_close(
        torch.as_tensor(ours(a.numpy(), b.numpy())), expected(a.numpy(), b.numpy())
    )


@pytest.mark.parametrize(
    ("ours", "st_similarity_fn"),
    [
        (pairwise_cos_sim, lambda util: util.pairwise_cos_sim),
        (pairwise_dot_score, lambda util: util.pairwise_dot_score),
        (pairwise_euclidean_sim, lambda util: util.pairwise_euclidean_sim),
    ],
)
def test_pairwise_similarity_matches_sentence_transformers(
    st_util: ModuleType,
    ours,
    st_similarity_fn,
    embeddings: tuple[torch.Tensor, torch.Tensor],
):
    a, b = embeddings
    b = b[: len(a)]  # one partner per row of `a`
    expected = st_similarity_fn(st_util)

    torch.testing.assert_close(torch.as_tensor(ours(a, b)), expected(a, b))
    torch.testing.assert_close(
        torch.as_tensor(ours(a.numpy(), b.numpy())), expected(a.numpy(), b.numpy())
    )
