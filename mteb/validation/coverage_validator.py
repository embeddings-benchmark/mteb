"""Reference model coverage validation for MTEB benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.load_results.benchmark_results import BenchmarkResults


class CoverageValidator:
    """Validates reference model coverage for MTEB benchmarks."""

    def __init__(self, benchmark: Benchmark):
        """Initialize validator for a specific benchmark.

        Args:
            benchmark: The benchmark to validate
        """
        self.benchmark = benchmark

    def get_required_reference_models(self) -> dict[str, list[str]]:
        """Get the reference models required for each task in this benchmark.

        Returns:
            Dictionary mapping task names to lists of required reference model names.
        """
        from mteb.reference_models import get_reference_models_for_task

        required_models = {}
        for task in self.benchmark.tasks:
            task_name = task.metadata.name
            required_models[task_name] = get_reference_models_for_task(task)

        return required_models

    def get_all_required_reference_models(self) -> list[str]:
        """Get all unique reference models required across all tasks in this benchmark.

        Returns:
            List of unique reference model names required for this benchmark.
        """
        required_by_task = self.get_required_reference_models()

        # Flatten and deduplicate
        all_required = set()
        for task_models in required_by_task.values():
            all_required.update(task_models)

        return sorted(list(all_required))

    def validate_reference_model_coverage(
        self, base_results: BenchmarkResults | None = None
    ) -> dict[str, list[str]]:
        """Validate that all reference models have results for this benchmark.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            Dictionary mapping task names to lists of missing reference model names.
        """
        from mteb.load_results import load_results

        try:
            if base_results is None:
                base_results = load_results()

            benchmark_results = self.benchmark.load_results(base_results)
            required_by_task = self.get_required_reference_models()
            missing_models = {}

            for task in self.benchmark.tasks:
                task_name = task.metadata.name
                required_models = required_by_task[task_name]

                # Get models that have results for this task
                task_results = benchmark_results.select_tasks([task])
                models_with_results = {
                    result.model_name for result in task_results.model_results
                }

                # Find missing reference models
                missing_for_task = [
                    model
                    for model in required_models
                    if model not in models_with_results
                ]

                missing_models[task_name] = missing_for_task

            return missing_models

        except Exception:
            # If we can't load results, return all models as missing
            required_by_task = self.get_required_reference_models()
            return {task: models for task, models in required_by_task.items()}

    def get_missing_reference_models(
        self, base_results: BenchmarkResults | None = None
    ) -> list[str]:
        """Get all missing reference models across all tasks in this benchmark.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            List of unique reference model names that are missing from any task.
        """
        missing_by_task = self.validate_reference_model_coverage(base_results)

        # Flatten and deduplicate
        all_missing = set()
        for task_missing in missing_by_task.values():
            all_missing.update(task_missing)

        return sorted(list(all_missing))

    def is_complete(self, base_results: BenchmarkResults | None = None) -> bool:
        """Check if this benchmark has complete reference model coverage.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            True if all reference models have results for all applicable tasks.
        """
        missing_models = self.get_missing_reference_models(base_results)
        return len(missing_models) == 0

    def get_reference_model_coverage_summary(
        self, base_results: BenchmarkResults | None = None
    ) -> dict[str, any]:
        """Get a comprehensive summary of reference model coverage.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            Dictionary with coverage statistics and details.
        """
        required_by_task = self.get_required_reference_models()
        missing_by_task = self.validate_reference_model_coverage(base_results)

        total_tasks = len(self.benchmark.tasks)

        # Calculate coverage statistics
        tasks_with_complete_coverage = sum(
            1 for missing in missing_by_task.values() if len(missing) == 0
        )

        unique_missing_models = set()
        for missing in missing_by_task.values():
            unique_missing_models.update(missing)

        unique_required_models = set()
        for required in required_by_task.values():
            unique_required_models.update(required)

        coverage_percentage = (
            (len(unique_required_models) - len(unique_missing_models))
            / len(unique_required_models)
            * 100
            if len(unique_required_models) > 0
            else 100
        )

        return {
            "benchmark_name": self.benchmark.name,
            "total_tasks": total_tasks,
            "total_reference_models": len(unique_required_models),
            "tasks_with_complete_coverage": tasks_with_complete_coverage,
            "tasks_with_missing_models": total_tasks - tasks_with_complete_coverage,
            "unique_missing_models": sorted(list(unique_missing_models)),
            "unique_required_models": sorted(list(unique_required_models)),
            "coverage_percentage": round(coverage_percentage, 2),
            "is_complete": len(unique_missing_models) == 0,
            "missing_by_task": missing_by_task,
            "required_by_task": required_by_task,
        }


# Convenience functions for direct use
def validate_benchmark_coverage(
    benchmark: Benchmark, base_results: BenchmarkResults | None = None
) -> dict[str, list[str]]:
    """Validate reference model coverage for a benchmark.

    Args:
        benchmark: The benchmark to validate
        base_results: Optional benchmark results to check against

    Returns:
        Dictionary mapping task names to lists of missing reference model names.
    """
    validator = CoverageValidator(benchmark)
    return validator.validate_reference_model_coverage(base_results)


def get_missing_reference_models(
    benchmark: Benchmark, base_results: BenchmarkResults | None = None
) -> list[str]:
    """Get missing reference models for a benchmark.

    Args:
        benchmark: The benchmark to validate
        base_results: Optional benchmark results to check against

    Returns:
        List of missing reference model names.
    """
    validator = CoverageValidator(benchmark)
    return validator.get_missing_reference_models(base_results)


def get_reference_model_coverage_summary(
    benchmark: Benchmark, base_results: BenchmarkResults | None = None
) -> dict[str, any]:
    """Get coverage summary for a benchmark.

    Args:
        benchmark: The benchmark to validate
        base_results: Optional benchmark results to check against

    Returns:
        Dictionary with coverage statistics and details.
    """
    validator = CoverageValidator(benchmark)
    return validator.get_reference_model_coverage_summary(base_results)
