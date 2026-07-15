"""Tests for the similarity primitives in ``mteb.similarity_functions``.

Relevance scores must be computed in float32 even when the encoder returns
low-precision (float16/bfloat16) embeddings, so that reduced precision does not
collapse distinct scores into spurious ties.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mteb.similarity_functions import (
    cos_sim,
    dot_score,
    pairwise_cos_sim,
    pairwise_dot_score,
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
