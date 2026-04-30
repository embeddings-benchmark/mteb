import numpy as np

import mteb
from mteb import ResultCache
from mteb.benchmarks._benchmark_metrics import (
    _compute_mean_task,
    _compute_mean_task_type,
)
from mteb.benchmarks.benchmark import Benchmark

MODELS_SCORES = {
    "Mean(Task)": {
        "mteb/baseline-random-encoder": 0.005604,
        "sentence-transformers/all-MiniLM-L6-v2": 0.59396733,
    },
    "Mean(TaskType)": {
        "mteb/baseline-random-encoder": 0.007336,
        "sentence-transformers/all-MiniLM-L6-v2": 0.645581,
    },
}


def _make_benchmark(extra_tasks: list[str] | None = None):
    tasks = mteb.get_tasks(
        [
            "NanoSCIDOCSRetrieval",
            "Banking77Classification",
            "NanoArguAnaRetrieval",
        ]
    )
    if extra_tasks:
        tasks += mteb.get_tasks(extra_tasks)
    return Benchmark(name="mock_benchmark", tasks=tasks)


def test_benchmark_get_score(mock_mteb_cache: ResultCache):
    """get_score returns Mean(Task), Mean(TaskType), and Borda Rank for each model."""
    mock_benchmark = _make_benchmark()
    mock_results = mock_mteb_cache.load_results()

    scores = mock_benchmark.get_score(mock_results)

    for model_name, expected in MODELS_SCORES["Mean(Task)"].items():
        assert model_name in scores, f"{model_name} missing from scores"
        assert np.allclose(scores[model_name]["Mean(Task)"], expected)

    for model_name, expected in MODELS_SCORES["Mean(TaskType)"].items():
        assert np.allclose(scores[model_name]["Mean(TaskType)"], expected)


def test_benchmark_get_score_missing_tasks(mock_mteb_cache: ResultCache):
    """get_score returns None for models with missing task results."""
    mock_benchmark = _make_benchmark(extra_tasks=["BelebeleRetrieval"])
    mock_results = mock_mteb_cache.load_results()

    scores = mock_benchmark.get_score(mock_results)

    for model_name in MODELS_SCORES["Mean(Task)"]:
        assert scores[model_name]["Mean(Task)"] is None
        assert scores[model_name]["Mean(TaskType)"] is None


def test_compute_mean_task(mock_mteb_cache: ResultCache):
    """_compute_mean_task returns the correct scalar mean."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert np.allclose(
        _compute_mean_task(task_results),
        MODELS_SCORES["Mean(Task)"][mock_model_name],
    )


def test_compute_mean_task_type(mock_mteb_cache: ResultCache):
    """_compute_mean_task_type returns the correct mean of task-type means."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert np.allclose(
        _compute_mean_task_type(task_results),
        MODELS_SCORES["Mean(TaskType)"][mock_model_name],
    )


def test_compute_mean_task_missing(mock_mteb_cache: ResultCache):
    """_compute_mean_task returns None when any task score is missing."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark(extra_tasks=["BelebeleRetrieval"])

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert _compute_mean_task(task_results) is None
    assert _compute_mean_task_type(task_results) is None
