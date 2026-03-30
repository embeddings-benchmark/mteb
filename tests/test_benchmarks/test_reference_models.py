"""Test that reference models have results for all public tasks in target benchmarks."""

import logging

import pytest

import mteb
from mteb import Benchmark, ResultCache
from mteb.abstasks import AbsTaskRetrieval
from mteb.leaderboard.benchmark_selector import (
    GP_BENCHMARK_ENTRIES,
    R_BENCHMARK_ENTRIES,
    MenuEntry,
)

logging.basicConfig(level=logging.INFO)

REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-multilingual-128M",
    "mteb/baseline-bm25s",
]

# Models that can only run retrieval tasks
RETRIEVAL_ONLY_MODELS = {"mteb/baseline-bm25s"}

# Benchmarks excluded from reference model testing.
# These are benchmarks displayed on the leaderboard (in benchmark_selector.py)
# for which reference models do not yet have results.
# When adding a new benchmark to the leaderboard, ensure reference models
# have results for it, or add it to this exclusion list.
EXCLUDED_BENCHMARKS = {
    # Multimodal benchmarks — reference models are text-only
    "HUME(v1)",
    "JinaVDR",
    "MAEB(beta)",
    "MAEB(beta, audio-only)",
    "MIEB(eng)",
    "MIEB(Img)",
    "MIEB(lite)",
    "MIEB(Multilingual)",
    "ViDoRe(v3)",
    "ViDoRe(v1&v2)",
}


def _collect_benchmarks(entries):
    """Recursively extract Benchmark objects from menu entries."""
    benchmarks = []
    for item in entries:
        if isinstance(item, Benchmark):
            benchmarks.append(item)
        elif isinstance(item, MenuEntry):
            benchmarks.extend(_collect_benchmarks(item.benchmarks))
    return benchmarks


def _get_target_benchmarks():
    """Get all benchmarks shown on the leaderboard minus excluded ones.

    Uses benchmark_selector.py as the source of truth for which benchmarks
    are displayed on the leaderboard, rather than display_on_leaderboard flag.
    """
    all_benchmarks = _collect_benchmarks(GP_BENCHMARK_ENTRIES + R_BENCHMARK_ENTRIES)
    return [b.name for b in all_benchmarks if b.name not in EXCLUDED_BENCHMARKS]


def _is_text_only_task(task):
    """Check if a task uses only text modalities."""
    return task.metadata.modalities == ["text"]


def _get_expected_task_names(benchmark, model_name):
    """Get public task names, filtering by model capabilities."""
    task_names = []
    for task in benchmark:
        if not task.metadata.is_public:
            continue
        if task.is_aggregate:
            task_names.append(task.metadata.name)
            continue
        if not _is_text_only_task(task):
            continue
        if model_name in RETRIEVAL_ONLY_MODELS and not isinstance(
            task, AbsTaskRetrieval
        ):
            continue
        task_names.append(task.metadata.name)
    return task_names


@pytest.fixture(scope="module")
def result_cache():
    cache = ResultCache()
    cache.download_from_remote()
    return cache


TARGET_BENCHMARKS = _get_target_benchmarks()


@pytest.mark.test_reference_models
@pytest.mark.parametrize("benchmark_name", TARGET_BENCHMARKS)
@pytest.mark.parametrize("model_name", REFERENCE_MODELS)
def test_reference_model_coverage(result_cache, benchmark_name, model_name):
    benchmark = mteb.get_benchmark(benchmark_name=benchmark_name)
    expected = _get_expected_task_names(benchmark, model_name)
    results = result_cache.load_results(models=[model_name])
    available = set(results.task_names)
    missing = sorted(set(expected) - available)
    if missing:
        pytest.fail(
            f"'{model_name}' missing {len(missing)}/{len(expected)} tasks "
            f"in '{benchmark_name}':\n" + "\n".join(f"  - {t}" for t in missing)
        )
