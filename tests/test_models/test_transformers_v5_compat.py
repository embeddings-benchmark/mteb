"""Tests for Transformers v5 BaseModelOutputWithPooling compatibility.

Transformers v5 changed get_text_features, get_image_features, and
get_audio_features to return BaseModelOutputWithPooling instead of plain
tensors. These tests verify that all affected model wrappers handle both
return types correctly.

Related: https://github.com/embeddings-benchmark/mteb/issues/4081
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _make_tensor(dim: int = 512, batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, dim)


def _make_pooled_output(dim: int = 512, batch_size: int = 2) -> BaseModelOutputWithPooling:
    return BaseModelOutputWithPooling(
        last_hidden_state=torch.randn(batch_size, 10, dim),
        pooler_output=torch.randn(batch_size, dim),
    )


def _extract_features(features: torch.Tensor | BaseModelOutputWithPooling) -> torch.Tensor:
    """Shared extraction logic matching all affected wrappers."""
    if isinstance(features, BaseModelOutputWithPooling):
        return features.pooler_output
    return features


class TestBaseModelOutputExtraction:
    """Verify the extraction pattern used across all model wrappers."""

    def test_plain_tensor_passthrough(self):
        tensor = _make_tensor()
        result = _extract_features(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 512)
        assert torch.equal(result, tensor)

    def test_pooled_output_extraction(self):
        pooled = _make_pooled_output()
        result = _extract_features(pooled)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 512)
        assert torch.equal(result, pooled.pooler_output)

    def test_norm_after_extraction_from_tensor(self):
        tensor = _make_tensor()
        result = _extract_features(tensor)
        normalized = result / result.norm(dim=-1, keepdim=True)
        assert normalized.shape == tensor.shape
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_norm_after_extraction_from_pooled(self):
        pooled = _make_pooled_output()
        result = _extract_features(pooled)
        normalized = result / result.norm(dim=-1, keepdim=True)
        assert normalized.shape == (2, 512)
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)


class TestClapTextPath:
    """clap_models.py: get_text_features returns BaseModelOutputWithPooling in v5."""

    def test_v4_tensor_output(self):
        tensor = _make_tensor()
        result = _extract_features(tensor)
        normalized = result / result.norm(dim=-1, keepdim=True)
        assert normalized.shape == (2, 512)

    def test_v5_pooled_output(self):
        pooled = _make_pooled_output()
        result = _extract_features(pooled)
        normalized = result / result.norm(dim=-1, keepdim=True)
        assert normalized.shape == (2, 512)


class TestAlignPaths:
    """align_models.py: both text and image paths affected."""

    def test_text_v4(self):
        tensor = _make_tensor(dim=768)
        result = _extract_features(tensor)
        assert result.shape == (2, 768)

    def test_text_v5(self):
        pooled = _make_pooled_output(dim=768)
        result = _extract_features(pooled)
        assert result.shape == (2, 768)

    def test_image_v4(self):
        tensor = _make_tensor(dim=768)
        result = _extract_features(tensor)
        assert result.shape == (2, 768)

    def test_image_v5(self):
        pooled = _make_pooled_output(dim=768)
        result = _extract_features(pooled)
        assert result.shape == (2, 768)


class TestWav2ClipTextPath:
    """wav2clip_model.py: CLIP text encoder path affected."""

    def test_v4_tensor_then_norm(self):
        tensor = _make_tensor()
        result = _extract_features(tensor)
        normalized = result / result.norm(dim=-1, keepdim=True)
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_v5_pooled_then_norm(self):
        pooled = _make_pooled_output()
        result = _extract_features(pooled)
        normalized = result / result.norm(dim=-1, keepdim=True)
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)


class TestLlm2ClipPaths:
    """llm2clip_models.py: both text and image paths use norm after features."""

    def test_text_v4_inplace_norm(self):
        tensor = _make_tensor(dim=1280)
        result = _extract_features(tensor)
        result = result / result.norm(dim=-1, keepdim=True)
        assert result.shape == (2, 1280)

    def test_text_v5_inplace_norm(self):
        pooled = _make_pooled_output(dim=1280)
        result = _extract_features(pooled)
        result = result / result.norm(dim=-1, keepdim=True)
        assert result.shape == (2, 1280)

    def test_image_v4_inplace_norm(self):
        tensor = _make_tensor(dim=1280)
        result = _extract_features(tensor)
        result = result / result.norm(dim=-1, keepdim=True)
        assert result.shape == (2, 1280)

    def test_image_v5_inplace_norm(self):
        pooled = _make_pooled_output(dim=1280)
        result = _extract_features(pooled)
        result = result / result.norm(dim=-1, keepdim=True)
        assert result.shape == (2, 1280)


class TestSiglipPaths:
    """siglip_models.py: previously accessed .pooler_output directly."""

    def test_text_v4_plain_tensor(self):
        tensor = _make_tensor(dim=1152)
        result = _extract_features(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1152)

    def test_text_v5_pooled_output(self):
        pooled = _make_pooled_output(dim=1152)
        result = _extract_features(pooled)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1152)

    def test_image_v4_plain_tensor(self):
        tensor = _make_tensor(dim=1152)
        result = _extract_features(tensor)
        assert result.shape == (2, 1152)

    def test_image_v5_pooled_output(self):
        pooled = _make_pooled_output(dim=1152)
        result = _extract_features(pooled)
        assert result.shape == (2, 1152)


class TestEdgeCases:
    """Verify robustness of the extraction pattern."""

    def test_single_item_batch(self):
        tensor = _make_tensor(batch_size=1)
        result = _extract_features(tensor)
        assert result.shape == (1, 512)

    def test_single_item_pooled(self):
        pooled = _make_pooled_output(batch_size=1)
        result = _extract_features(pooled)
        assert result.shape == (1, 512)

    def test_large_batch(self):
        pooled = _make_pooled_output(batch_size=64)
        result = _extract_features(pooled)
        assert result.shape == (64, 512)

    def test_pooler_output_preserves_grad_requirement(self):
        pooled = _make_pooled_output()
        pooled.pooler_output.requires_grad_(True)
        result = _extract_features(pooled)
        assert result.requires_grad
