from importlib.metadata import version

from mteb import types
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.deprecated_evaluator import MTEB
from mteb.evaluate import evaluate
from mteb.filter_tasks import filter_tasks
from mteb.get_tasks import get_task, get_tasks
from mteb.load_results import load_results
from mteb.models import (
    CacheBackendProtocol,
    CrossEncoderProtocol,
    EncoderProtocol,
    IndexEncoderSearchProtocol,
    SearchProtocol,
    SentenceTransformerEncoderWrapper,
)
from mteb.models.get_model_meta import get_model, get_model_meta, get_model_metas
from mteb.results import BenchmarkResults, TaskResult

from .benchmarks.benchmark import Benchmark
from .benchmarks.get_benchmark import get_benchmark, get_benchmarks

__version__ = version("mteb")  # fetch version from install metadata

__all__ = [
    "MTEB",
    "AbsTask",
    "Benchmark",
    "BenchmarkResults",
    "CacheBackendProtocol",
    "CrossEncoderProtocol",
    "EncoderProtocol",
    "IndexEncoderSearchProtocol",
    "SearchProtocol",
    "SentenceTransformerEncoderWrapper",
    "TaskMetadata",
    "TaskResult",
    "evaluate",
    "filter_tasks",
    "get_benchmark",
    "get_benchmarks",
    "get_model",
    "get_model_meta",
    "get_model_metas",
    "get_task",
    "get_tasks",
    "load_results",
    "types",
]
