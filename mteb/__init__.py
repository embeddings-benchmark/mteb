from importlib.metadata import version

from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.evaluate import evaluate
from mteb.load_results import load_results
from mteb.models import Encoder, SentenceTransformerEncoderWrapper
from mteb.models.get_model_meta import get_model, get_model_meta, get_model_metas
from mteb.MTEB import MTEB
from mteb.overview import get_task, get_tasks
from mteb.results import BenchmarkResults, TaskResult

from .benchmarks.benchmark import Benchmark
from .benchmarks.get_benchmark import get_benchmark, get_benchmarks

__version__ = version("mteb")  # fetch version from install metadata

__all__ = [
    "MTEB",
    "AbsTask",
    "Benchmark",
    "BenchmarkResults",
    "Encoder",
    "SentenceTransformerEncoderWrapper",
    "TaskMetadata",
    "TaskResult",
    "evaluate",
    "get_benchmark",
    "get_benchmarks",
    "get_model",
    "get_model_meta",
    "get_model_metas",
    "get_task",
    "get_tasks",
    "load_results",
]
