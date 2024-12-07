from __future__ import annotations

from importlib.metadata import version

from mteb.abstasks import AbsTask, TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.evaluation import TaskResult
from mteb.evaluation.MTEB import MTEB
from mteb.load_results import BenchmarkResults, load_results
from mteb.models import get_model, get_model_meta, get_model_metas
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks

from .benchmarks.benchmarks import Benchmark
from .benchmarks.get_benchmark import BENCHMARK_REGISTRY, get_benchmark, get_benchmarks

__version__ = version("mteb")  # fetch version from install metadata


__all__ = [
    "get_tasks",
    "get_task",
    "get_model",
    "get_model_meta",
    "get_model_metas",
    "load_results",
    "Benchmark",
    "get_benchmark",
    "get_benchmarks",
    "BenchmarkResults",
    "BENCHMARK_REGISTRY",
    "MTEB",
    "TASKS_REGISTRY",
    "TaskResult",
    "TaskMetadata",
    "Encoder",
    "AbsTask",
]
