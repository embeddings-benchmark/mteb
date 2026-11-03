from __future__ import annotations

import difflib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from .benchmarks import BENCHMARK_REGISTRY

if TYPE_CHECKING:
    from .benchmark import Benchmark

logger = logging.getLogger(__name__)


@lru_cache
def _build_aliases_registry() -> dict[str, Benchmark]:
    aliases: dict[str, Benchmark] = {}
    for inst in BENCHMARK_REGISTRY.values():
        if inst.aliases is not None:
            for alias in inst.aliases:
                aliases[alias] = inst
    return aliases


def get_benchmark(
    benchmark_name: str,
) -> Benchmark:
    """Get a benchmark by name.

    Args:
        benchmark_name: The name of the benchmark to retrieve.

    Returns:
        The Benchmark instance corresponding to the given name.
    """
    benchmark_registry = BENCHMARK_REGISTRY
    aliases_registry = _build_aliases_registry()

    if benchmark_name in aliases_registry:
        return aliases_registry[benchmark_name]
    if benchmark_name not in benchmark_registry:
        close_matches = difflib.get_close_matches(
            benchmark_name, benchmark_registry.keys()
        )
        if close_matches:
            suggestion = (
                f"'{benchmark_name}' not found. Did you mean: {close_matches[0]}?"
            )
        else:
            suggestion = f"'{benchmark_name}' not found and no similar keys were found."
        raise KeyError(suggestion)
    return benchmark_registry[benchmark_name]


def get_benchmarks(
    names: list[str] | None = None, display_on_leaderboard: bool | None = None
) -> list[Benchmark]:
    """Get a list of benchmarks by name.

    Args:
        names: A list of benchmark names to retrieve. If None, all benchmarks are returned.
        display_on_leaderboard: If specified, filters benchmarks by whether they are displayed on the leaderboard.

    Returns:
        A list of Benchmark instances.
    """
    benchmark_registry = BENCHMARK_REGISTRY

    if names is None:
        names = list(benchmark_registry.keys())
    benchmarks = [get_benchmark(name) for name in names]
    if display_on_leaderboard is not None:
        benchmarks = [
            b for b in benchmarks if b.display_on_leaderboard is display_on_leaderboard
        ]
    return benchmarks
