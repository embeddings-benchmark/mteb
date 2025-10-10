"""Tests for reference model integration with existing MTEB tooling."""

from __future__ import annotations

import sys
from pathlib import Path


def test_check_results_script_uses_reference_models():
    """Test that check_results.py script uses the reference models registry."""
    # Add the scripts directory to path
    scripts_dir = Path(__file__).parent.parent / "scripts" / "running_model"
    sys.path.insert(0, str(scripts_dir))

    try:
        import check_results

        # Should have imported reference models
        assert hasattr(
            check_results, "reference_models"
        ), "check_results.py should import reference models"
        assert hasattr(
            check_results, "model_names"
        ), "check_results.py should have model_names list"

        # Reference models should be in the model list
        from mteb.reference_models import get_reference_models

        reference_models = get_reference_models()

        for ref_model in reference_models:
            assert (
                ref_model in check_results.model_names
            ), f"Reference model {ref_model} should be in check_results model list"

        for i, ref_model in enumerate(reference_models):
            assert (
                check_results.model_names[i] == ref_model
            ), f"Reference model {ref_model} should be at position {i}"

    finally:
        # Clean up sys.path
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))


def test_backward_compatibility():
    """Test that adding reference models doesn't break existing functionality."""
    import mteb

    # Basic MTEB functionality should still work
    tasks = mteb.get_tasks(task_types=["Classification"], languages=["eng"])[:2]
    assert len(tasks) > 0, "Should still be able to get tasks"

    benchmarks = mteb.get_benchmarks()
    assert len(benchmarks) > 0, "Should still be able to get benchmarks"

    # Model loading should still work
    meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
    assert meta.name == "sentence-transformers/all-MiniLM-L6-v2"


def test_reference_models_do_not_interfere_with_existing_models():
    """Test that reference models don't interfere with existing model functionality."""
    import mteb

    # Non-reference models should still work normally
    non_ref_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Should be able to get metadata
    meta = mteb.get_model_meta(non_ref_model)
    assert meta.name == non_ref_model

    # Should not be identified as reference model
    assert not mteb.is_reference_model(non_ref_model)

    # Reference model check should work correctly
    assert mteb.is_reference_model("intfloat/multilingual-e5-small")


def test_validation_module_integration():
    """Test that validation module integrates properly with MTEB."""
    import mteb

    # Should be able to import validation functions from mteb
    assert hasattr(mteb, "CoverageValidator")
    assert hasattr(mteb, "get_missing_reference_models")

    # Should work with actual benchmarks
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")

    # Should be able to create validator
    validator = mteb.CoverageValidator(benchmark)
    assert validator.benchmark == benchmark

    # Should be able to use standalone functions
    try:
        missing = mteb.get_missing_reference_models(benchmark)
        assert isinstance(missing, list)
    except Exception:
        # Expected if no results available
        pass
