from __future__ import annotations

from mteb.benchmarks.benchmark import Benchmark
from mteb.benchmarks.benchmarks import *
from mteb.benchmarks.get_benchmark import (
    BENCHMARK_REGISTRY,
    get_benchmark,
    get_benchmarks,
)
from mteb.benchmarks.rteb_benchmarks import *

__all__ = [
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "get_benchmarks",
    "Benchmark",
]
