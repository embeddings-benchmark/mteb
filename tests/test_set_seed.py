"""Tests for `mteb._set_seed`."""

from __future__ import annotations


def test_set_seed_still_seeds_torch_when_available() -> None:
    """Torch must still be seeded once it is imported.

    `_set_seed` runs when tasks are created, which happens before torch is imported, so it only seeds torch
    when torch is already there. Evaluators seed again when they are created, after the model is loaded.
    """
    import torch

    from mteb._set_seed import _set_seed

    _set_seed(42)
    first = torch.randn(4)
    _set_seed(42)
    second = torch.randn(4)
    assert torch.equal(first, second)
