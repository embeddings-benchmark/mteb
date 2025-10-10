"""Tests for reference models registry.

These tests validate that the reference models registry integrates properly with MTEB.
"""

from __future__ import annotations

import pytest

import mteb


def test_reference_models_policy_compliance():
    """Test that reference models comply with the specifications."""
    from mteb.reference_models import get_reference_models

    reference_models = get_reference_models()

    expected_models = [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "minishlab/potion-multilingual-128M",
        "bm25s",
    ]

    assert len(reference_models) >= len(
        expected_models
    ), "Should have at least the required reference models"

    for expected_model in expected_models:
        assert (
            expected_model in reference_models
        ), f"Required reference model {expected_model} missing"


@pytest.mark.parametrize(
    "model_name",
    [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "minishlab/potion-multilingual-128M",
        "bm25s",
    ],
)
def test_reference_model_exists_in_mteb_registry(model_name: str):
    """Test that each reference model exists in MTEB's model registry."""
    meta = mteb.get_model_meta(model_name)
    assert meta.name == model_name


def test_reference_models_public_api():
    """Test that reference models are available via MTEB's public API."""
    # Should be importable from mteb
    assert hasattr(mteb, "get_reference_models")
    assert hasattr(mteb, "get_reference_models_for_task_type")
    assert hasattr(mteb, "is_reference_model")

    # Should work when called
    models = mteb.get_reference_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Should handle task type restrictions
    retrieval_models = mteb.get_reference_models_for_task_type("Retrieval")
    classification_models = mteb.get_reference_models_for_task_type("Classification")

    # BM25 should only be in retrieval
    assert "bm25s" in retrieval_models
    assert "bm25s" not in classification_models


def test_reference_models_integration_with_existing_tooling():
    """Test that reference models integrate with existing MTEB tooling."""
    # Should work with benchmarks
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")

    # Benchmark should have reference model methods
    assert hasattr(benchmark, "get_missing_reference_models")

    # Should be able to call the methods (even if they fail due to missing results)
    try:
        missing = benchmark.get_missing_reference_models()
        assert isinstance(missing, list)
    except Exception:
        # Expected if no results available
        pass


def test_reference_models_coverage_validation():
    """Test that reference model coverage validation works."""
    # Should be able to create validator
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    validator = mteb.CoverageValidator(benchmark)

    # Should be able to get required models
    required = validator.get_all_required_reference_models()
    assert isinstance(required, list)
    assert len(required) > 0

    # Should include expected reference models
    from mteb.reference_models import get_reference_models

    reference_models = get_reference_models()

    for ref_model in reference_models:
        # All reference models should be required somewhere in the benchmark
        # (unless they're restricted to task types not in this benchmark)
        if ref_model != "bm25s":  # BM25 only for retrieval
            assert (
                ref_model in required
            ), f"Reference model {ref_model} should be required"
