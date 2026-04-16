"""Test that reference models have results for all public tasks in target benchmarks."""

import logging

import pytest

import mteb
from mteb import BenchmarkResults, ResultCache
from mteb.abstasks import AbsTaskRetrieval
from mteb.models.get_model_meta import get_model_meta

logging.basicConfig(level=logging.INFO)

REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-multilingual-128M",
    "mteb/baseline-bm25s",
]


def _is_task_compatible(task, model_meta):
    """Check if a task is compatible with the model based on modalities."""
    if model_meta.modalities is None:
        return True
    model_mods = set(model_meta.modalities)
    task_mods = set(task.metadata.modalities)
    # For retrieval tasks, check query and document modalities separately
    if isinstance(task, AbsTaskRetrieval):
        from mteb.types import PromptType

        query_mods = set(task.metadata.get_modalities(PromptType.query))
        doc_mods = set(task.metadata.get_modalities(PromptType.document))
        return bool(model_mods & query_mods) and bool(model_mods & doc_mods)
    return task_mods.issubset(model_mods)


def _get_expected_task_names(benchmark, model_name):
    """Get public task names compatible with the model's modalities."""
    model_meta = get_model_meta(model_name)
    task_names = []
    benchmark_tasks = []
    for task in benchmark.tasks:
        if task.is_aggregate:
            benchmark_tasks.extend(task.metadata.tasks)
        else:
            benchmark_tasks.append(task)
    for task in benchmark_tasks:
        if not task.metadata.is_public:
            continue
        if not _is_task_compatible(task, model_meta):
            continue
        task_names.append(task.metadata.name)
    return task_names


@pytest.fixture(scope="module")
def result_cache() -> BenchmarkResults:
    cache = ResultCache()
    cache.download_from_remote()
    results_cache = cache.load_results(models=REFERENCE_MODELS)
    return results_cache


TARGET_BENCHMARKS = mteb.get_benchmarks(display_on_leaderboard=True)


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
