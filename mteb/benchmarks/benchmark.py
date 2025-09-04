from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter

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
