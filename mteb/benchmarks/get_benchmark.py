from __future__ import annotations

import difflib

import mteb.benchmarks.benchmarks as benchmark_module
from mteb.benchmarks.benchmarks import Benchmark

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
    names: list[str] | None = None,
) -> list[Benchmark]:
    if names is None:
        names = list(BENCHMARK_REGISTRY.keys())
    return [BENCHMARK_REGISTRY[name] for name in names]
