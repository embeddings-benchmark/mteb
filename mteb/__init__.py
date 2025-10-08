from __future__ import annotations

from importlib.metadata import version

from mteb.benchmarks.benchmarks import (
    MTEB_ENG_CLASSIC,
    MTEB_MAIN_RU,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_MEDICAL,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
    CoIR,
)
from mteb.evaluation import *
from mteb.load_results import BenchmarkResults, load_results
from mteb.models import get_model, get_model_meta, get_model_metas
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks
from mteb.reference_models import (
    get_reference_models,
    get_reference_models_for_task_type,
    get_reference_models_for_task,
    is_reference_model,
)
from mteb.validation import (
    CoverageValidator,
    validate_benchmark_coverage,
    get_missing_reference_models,
    get_reference_model_coverage_summary,
)

from .benchmarks.benchmarks import Benchmark
from .benchmarks.get_benchmark import BENCHMARK_REGISTRY, get_benchmark, get_benchmarks

__version__ = version("mteb")  # fetch version from install metadata


__all__ = [
    "MTEB_ENG_CLASSIC",
    "MTEB_MAIN_RU",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_MEDICAL",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "CoIR",
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
    "BENCHMARK_REGISTRY",
    "get_reference_models",
    "get_reference_models_for_task_type",
    "get_reference_models_for_task",
    "is_reference_model",
    "CoverageValidator",
    "validate_benchmark_coverage",
    "get_missing_reference_models",
    "get_reference_model_coverage_summary",
]
