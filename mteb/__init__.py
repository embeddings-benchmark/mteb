from __future__ import annotations

from importlib.metadata import version

from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.evaluate import evaluate
from mteb.load_results import BenchmarkResults, load_results
from mteb.load_results.task_results import TaskResult
from mteb.models import Encoder, SentenceTransformerEncoderWrapper
from mteb.models.get_model_meta import get_model, get_model_meta, get_model_metas
from mteb.MTEB import MTEB
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks

from .benchmarks.benchmark import Benchmark
from .benchmarks.get_benchmark import get_benchmark, get_benchmarks

__version__ = version("mteb")  # fetch version from install metadata

__all__ = [
    "TASKS_REGISTRY",
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
    "MTEB",
    "TaskResult",
    "TaskMetadata",
    "Encoder",
    "AbsTask",
    "SentenceTransformerEncoderWrapper",
    "evaluate",
]
