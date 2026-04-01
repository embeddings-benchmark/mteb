"""Test that reference models have results for all public tasks in target benchmarks."""

import logging

import pytest

import mteb
from mteb import BenchmarkResults, ResultCache
from mteb.abstasks import AbsTaskRetrieval

logging.basicConfig(level=logging.INFO)

REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-multilingual-128M",
    "mteb/baseline-bm25s",
]

# Models that can only run retrieval tasks
RETRIEVAL_ONLY_MODELS = {"mteb/baseline-bm25s"}


def _get_target_benchmarks():
    """Get all benchmarks displayed on the leaderboard.

    Uses display_on_leaderboard flag, which is kept in sync with
    benchmark_selector.py (see PR #4288).
    Task-level filtering (text-only, retrieval-only) is handled in
    _get_expected_task_names, so no benchmark-level exclusions are needed.
    """
    return mteb.get_benchmarks(display_on_leaderboard=True)


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
def result_cache() -> BenchmarkResults:
    cache = ResultCache()
    cache.download_from_remote()
    results_cache = cache.load_results(models=REFERENCE_MODELS)
    return results_cache


TARGET_BENCHMARKS = _get_target_benchmarks()


@pytest.mark.test_reference_models
@pytest.mark.parametrize("benchmark", TARGET_BENCHMARKS, ids=lambda b: b.name)
@pytest.mark.parametrize("model_name", REFERENCE_MODELS)
def test_reference_model_coverage(result_cache, benchmark, model_name):
    expected = _get_expected_task_names(benchmark, model_name)
    results = result_cache._filter_models(model_names=[model_name])._filter_tasks(
        expected
    )
    available = set(results.task_names)
    missing = sorted(set(expected) - available)
    if missing:
        pytest.fail(
            f"'{model_name}' missing {len(missing)}/{len(expected)} tasks "
            f"in '{benchmark.name}':\n" + "\n".join(f"  - {t}" for t in missing)
        )
