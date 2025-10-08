from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import pandas as pd
from pydantic import AnyUrl, BeforeValidator, TypeAdapter

from mteb.benchmarks._create_table import (
    _create_per_task_table_from_benchmark_results,
    _create_summary_table_from_benchmark_results,
    _create_summary_table_mean_public_private,
    _create_summary_table_mean_subset,
    _create_summary_table_mean_task_type,
)
from mteb.load_results.load_results import load_results

if TYPE_CHECKING:
    from mteb.abstasks.AbsTask import AbsTask
    from mteb.load_results.benchmark_results import BenchmarkResults

http_url_adapter = TypeAdapter(AnyUrl)
UrlString = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL


@dataclass
class Benchmark:
    """A benchmark object intended to run a certain benchmark within MTEB.

    Args:
        name: The name of the benchmark
        tasks: The tasks within the benchmark.
        description: A description of the benchmark, should include its intended goal and potentially a description of its construction
        reference: A link reference, to a source containing additional information typically to a paper, leaderboard or github.
        citation: A bibtex citation
        contacts: The people to contact in case of a problem in the benchmark, preferably a GitHub handle.

    Example:
        >>> Benchmark(
        ...     name="MTEB(custom)",
        ...     tasks=mteb.get_tasks(
        ...         tasks=["AmazonCounterfactualClassification", "AmazonPolarityClassification"],
        ...         languages=["eng"],
        ...     ),
        ...     description="A custom benchmark"
        ... )
    """

    name: str
    tasks: Sequence[AbsTask]
    description: str | None = None
    reference: UrlString | None = None
    citation: str | None = None
    contacts: list[str] | None = None
    display_on_leaderboard: bool = True
    icon: str | None = None
    display_name: str | None = None

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    def load_results(
        self, base_results: None | BenchmarkResults = None
    ) -> BenchmarkResults:
        if not hasattr(self, "results_cache"):
            self.results_cache = {}
        if base_results in self.results_cache:
            return self.results_cache[base_results]
        if base_results is None:
            base_results = load_results()
        results = base_results.select_tasks(self.tasks)
        self.results_cache[base_results] = results
        return results

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app."""
        return _create_summary_table_from_benchmark_results(benchmark_results)

    def _create_per_task_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create per-task table. Called by the leaderboard app."""
        return _create_per_task_table_from_benchmark_results(benchmark_results)

    def get_required_reference_models(self) -> dict[str, list[str]]:
        """Get the reference models required for each task in this benchmark.

        This is a method that determines what models should be evaluated.

        Returns:
            Dictionary mapping task names to lists of required reference model names.
        """
        from mteb.reference_models import get_reference_models_for_task

        required_models = {}
        for task in self.tasks:
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
        """Validate reference model coverage for this benchmark.

        This is a convenience method that delegates to the validation module.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            Dictionary mapping task names to lists of missing reference model names.
        """
        from mteb.validation import CoverageValidator

        validator = CoverageValidator(self)
        return validator.validate_reference_model_coverage(base_results)

    def get_missing_reference_models(
        self, base_results: BenchmarkResults | None = None
    ) -> list[str]:
        """Get missing reference models for this benchmark.

        This is a convenience method that delegates to the validation module.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            List of missing reference model names.
        """
        from mteb.validation import CoverageValidator

        validator = CoverageValidator(self)
        return validator.get_missing_reference_models(base_results)

    def get_reference_model_coverage_summary(
        self, base_results: BenchmarkResults | None = None
    ) -> dict[str, any]:
        """Get reference model coverage summary for this benchmark.

        This is a convenience method that delegates to the validation module.

        Args:
            base_results: Optional benchmark results to check against.

        Returns:
            Dictionary with coverage statistics and details.
        """
        from mteb.validation import CoverageValidator

        validator = CoverageValidator(self)
        return validator.get_reference_model_coverage_summary(base_results)


class RtebBenchmark(Benchmark):
    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app."""
        return _create_summary_table_mean_public_private(benchmark_results)


class HUMEBenchmark(Benchmark):
    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app."""
        return _create_summary_table_mean_subset(benchmark_results)


class MIEBBenchmark(Benchmark):
    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app."""
        return _create_summary_table_mean_task_type(benchmark_results)
