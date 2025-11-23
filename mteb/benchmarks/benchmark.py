from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from mteb.benchmarks._create_table import (
    _create_per_task_table_from_benchmark_results,
    _create_summary_table_from_benchmark_results,
    _create_summary_table_mean_public_private,
    _create_summary_table_mean_subset,
    _create_summary_table_mean_task_type,
)
from mteb.results import BenchmarkResults
from mteb.types import StrURL

if TYPE_CHECKING:
    from mteb.abstasks import AbsTask


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

    Examples:
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
    tasks: Sequence["AbsTask"]
    description: str | None = None
    reference: StrURL | None = None
    citation: str | None = None
    contacts: list[str] | None = None
    display_on_leaderboard: bool = True
    icon: str | None = None
    display_name: str | None = None

    def __iter__(self) -> Iterable["AbsTask"]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> "AbsTask":
        return self.tasks[index]

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app.

        Returns:
            A pandas DataFrame representing the summary results.
        """
        return _create_summary_table_from_benchmark_results(benchmark_results)

    def _create_per_task_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create per-task table. Called by the leaderboard app.

        Returns:
            A pandas DataFrame representing the per-task results.
        """
        return _create_per_task_table_from_benchmark_results(benchmark_results)


class RtebBenchmark(Benchmark):
    """Wrapper for RTEB benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        joint_table = _create_summary_table_mean_public_private(benchmark_results)
        # For RTEB: all tasks are Retrieval type, so Retrieval column = Mean (Task)
        joint_table = joint_table.rename(columns={"Retrieval": "Mean (Task)"})
        return joint_table


class HUMEBenchmark(Benchmark):
    """Wrapper for HUME benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        return _create_summary_table_mean_subset(benchmark_results)


class MIEBBenchmark(Benchmark):
    """Wrapper for MIEB benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        return _create_summary_table_mean_task_type(benchmark_results)


class VidoreBenchmark(Benchmark):
    """Wrapper for Vidore3 benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        joint_table = _create_summary_table_mean_public_private(benchmark_results)
        # For ViDoRe (V1, V2, V3): all tasks are Document Understanding type, so Document Understanding column = Mean (Task)
        joint_table = joint_table.rename(
            columns={"Document Understanding": "Mean (Task)"}
        )
        return joint_table
