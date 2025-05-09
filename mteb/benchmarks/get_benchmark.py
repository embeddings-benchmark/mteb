from __future__ import annotations

import difflib
import logging

import mteb.benchmarks.benchmarks as benchmark_module

from .benchmark import Benchmark

logger = logging.getLogger(__name__)

BENCHMARK_REGISTRY = {
    inst.name: inst
    for nam, inst in benchmark_module.__dict__.items()
    if isinstance(inst, Benchmark)
}


def get_benchmark(
    benchmark_name: str,
) -> Benchmark:
    if benchmark_name not in BENCHMARK_REGISTRY:
        close_matches = difflib.get_close_matches(
            benchmark_name, BENCHMARK_REGISTRY.keys()
        )
        if close_matches:
            suggestion = f"KeyError: '{benchmark_name}' not found. Did you mean: {close_matches[0]}?"
        else:
            suggestion = f"KeyError: '{benchmark_name}' not found and no similar keys were found."
        raise KeyError(suggestion)
    return BENCHMARK_REGISTRY[benchmark_name]


def get_benchmarks(
    names: list[str] | None = None, display_on_leaderboard: bool | None = None
) -> list[Benchmark]:
    if names is None:
        names = list(BENCHMARK_REGISTRY.keys())
    benchmarks = [get_benchmark(name) for name in names]
    if display_on_leaderboard is not None:
        benchmarks = [
            b for b in benchmarks if b.display_on_leaderboard is display_on_leaderboard
        ]
    return benchmarks
