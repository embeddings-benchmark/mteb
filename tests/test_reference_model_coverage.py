"""Test that ensures reference models are evaluated on all relevant benchmarks.

"Implementation of a test that ensures results for all benchmarks"

The test will FAIL if reference models are missing from benchmarks,
enforcing the policy that reference models must be evaluated.
"""

from __future__ import annotations

import pytest

import mteb
from mteb.reference_models import get_reference_models_for_task


class TestReferenceModelCoverage:
    """Test that reference models are properly evaluated on benchmarks."""

    def test_reference_models_exist_in_registry(self):
        """Test that all reference models are properly registered in MTEB."""
        from mteb.reference_models import get_reference_models
        
        reference_models = get_reference_models()
        
        # Should have the reference models
        expected_models = [
            "intfloat/multilingual-e5-small",
            "sentence-transformers/static-similarity-mrl-multilingual-v1",
            "minishlab/potion-multilingual-128M",
            "bm25s",
        ]
        
        for model_name in expected_models:
            assert model_name in reference_models, f"Reference model {model_name} missing from registry"
            
            # Verify model exists in MTEB registry
            try:
                meta = mteb.get_model_meta(model_name)
                assert meta.name == model_name
            except Exception as e:
                pytest.fail(f"Reference model {model_name} not found in MTEB registry: {e}")

    @pytest.mark.parametrize("benchmark_name", [
        "MTEB(eng, v2)",
        # We can add more benchmarks as they get reference model coverage
    ])
    def test_benchmark_reference_model_coverage(self, benchmark_name: str):
        """Test that benchmark has reference model coverage.
        
        It will FAIL if reference models haven't been evaluated on the benchmark.
        
        This enforces the policy that reference models should be evaluated
        on all benchmarks before they are considered complete.
        """
        benchmark = mteb.get_benchmark(benchmark_name)
        
        # Load results to check what models have been evaluated
        try:
            results = benchmark.load_results()
        except Exception:
            pytest.skip(f"No results available for benchmark {benchmark_name} - cannot check coverage")
        
        # Get all models that have results for this benchmark
        models_with_results = {result.model_name for result in results.model_results}
        
        if not models_with_results:
            pytest.skip(f"No model results found for benchmark {benchmark_name}")
        
        # Check each task in the benchmark
        missing_coverage = {}
        
        for task in benchmark.tasks:
            task_name = task.metadata.name
            required_models = get_reference_models_for_task(task)
            
            # Find which reference models are missing results for this task
            task_results = results.select_tasks([task])
            task_models = {result.model_name for result in task_results.model_results}
            
            missing_models = [
                model for model in required_models 
                if model not in task_models
            ]
            
            if missing_models:
                missing_coverage[task_name] = missing_models
        
        # If any reference models are missing, fail the test
        if missing_coverage:
            error_msg = f"Reference model coverage incomplete for benchmark {benchmark_name}:\n\n"
            
            for task_name, missing_models in missing_coverage.items():
                error_msg += f" Task: {task_name}\n"
                error_msg += f" Missing models: {missing_models}\n\n"
            
            error_msg += "   To fix this:\n"
            error_msg += "   1. Evaluate the missing reference models on the missing tasks\n"
            error_msg += "   2. Add the results to the results repository\n"
            error_msg += "   3. Re-run this test\n\n"
            error_msg += "   This test enforces that all reference models\n"
            error_msg += "   should be evaluated on all relevant benchmarks.\n"
            
            pytest.fail(error_msg)

    def test_reference_model_task_type_restrictions(self):
        """Test that reference models are correctly restricted by task type."""
        from mteb.reference_models import get_reference_models_for_task_type
        
        # BM25 should only be required for retrieval tasks
        retrieval_models = get_reference_models_for_task_type("Retrieval")
        classification_models = get_reference_models_for_task_type("Classification")
        
        assert "bm25s" in retrieval_models, "BM25 should be required for retrieval tasks"
        assert "bm25s" not in classification_models, "BM25 should not be required for classification tasks"
        
        # Other models should be required for all task types
        common_models = [
            "intfloat/multilingual-e5-small",
            "sentence-transformers/static-similarity-mrl-multilingual-v1",
            "minishlab/potion-multilingual-128M",
        ]
        
        for model in common_models:
            assert model in retrieval_models, f"{model} should be required for retrieval tasks"
            assert model in classification_models, f"{model} should be required for classification tasks"

    @pytest.mark.parametrize("benchmark_name", [
        "MTEB(eng, v2)",
        "MTEB(eng, v1)",
    ])
    def test_benchmark_has_reference_model_requirements(self, benchmark_name: str):
        """Test that benchmark correctly identifies required reference models."""
        benchmark = mteb.get_benchmark(benchmark_name)
        
        # Should be able to get required reference models for each task
        from mteb.validation import CoverageValidator
        validator = CoverageValidator(benchmark)
        
        required_by_task = validator.get_required_reference_models()
        
        assert isinstance(required_by_task, dict)
        assert len(required_by_task) > 0
        
        # Should have entries for all tasks in benchmark
        task_names = {task.metadata.name for task in benchmark.tasks}
        assert set(required_by_task.keys()) == task_names
        
        # Each task should have some required models
        for task_name, required_models in required_by_task.items():
            assert isinstance(required_models, list)
            assert len(required_models) > 0, f"Task {task_name} should have required reference models"
