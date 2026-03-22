"""Test that reference models have results for all public tasks in target benchmarks."""

import logging

import pytest

import mteb
from mteb import ResultCache

logging.basicConfig(level=logging.INFO)

REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-multilingual-128M",
    "mteb/baseline-bm25s",
]

TARGET_BENCHMARKS = [
    "MTEB(Multilingual, v2)",
    "MTEB(eng, v2)",
    "MTEB(Europe, v1)",
    "MTEB(Indic, v1)",
    "MTEB(Scandinavian, v1)",
    "RTEB(beta)",
]

# Models that can only run retrieval tasks
RETRIEVAL_ONLY_MODELS = {"mteb/baseline-bm25s"}


def _get_expected_task_names(benchmark, model_name):
    """Get public task names, filtering by model capabilities."""
    task_names = []
    for task in benchmark:
        if not task.metadata.is_public:
            continue
        if task.is_aggregate:
            task_names.append(task.metadata.name)
        else:
            if (
                model_name in RETRIEVAL_ONLY_MODELS
                and task.metadata.type != "Retrieval"
            ):
                continue
            task_names.append(task.metadata.name)
    return task_names


def _get_available_tasks_for_model(cache_path, model_name):
    """Get all task names with results across all revisions."""
    model_dir = model_name.replace("/", "__").replace(" ", "_")
    model_path = cache_path / "remote" / "results" / model_dir
    if not model_path.exists():
        return set()
    found = set()
    for rev_dir in model_path.iterdir():
        if rev_dir.is_dir():
            for f in rev_dir.glob("*.json"):
                if f.name != "model_meta.json":
                    found.add(f.stem)
    return found


@pytest.fixture(scope="module")
def result_cache():
    cache = ResultCache()
    cache.download_from_remote()
    return cache


@pytest.mark.test_reference_models
@pytest.mark.parametrize("benchmark_name", TARGET_BENCHMARKS)
@pytest.mark.parametrize("model_name", REFERENCE_MODELS)
def test_reference_model_coverage(result_cache, benchmark_name, model_name):
    benchmark = mteb.get_benchmark(benchmark_name=benchmark_name)
    expected = _get_expected_task_names(benchmark, model_name)
    available = _get_available_tasks_for_model(result_cache.cache_path, model_name)
    missing = sorted(set(expected) - available)
    if missing:
        pytest.fail(
            f"'{model_name}' missing {len(missing)}/{len(expected)} tasks "
            f"in '{benchmark_name}':\n" + "\n".join(f"  - {t}" for t in missing)
        )
