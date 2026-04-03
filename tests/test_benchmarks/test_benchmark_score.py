import numpy as np
import pytest

import mteb
from mteb import ResultCache
from mteb.benchmarks.benchmark import AggregationLevel, Benchmark


@pytest.mark.parametrize(
    ("aggregation_level", "models_scores"),
    [
        (
            AggregationLevel.mean_per_task,
            {
                "mteb/baseline-random-encoder": 0.005604,
                "sentence-transformers/all-MiniLM-L6-v2": 0.59396733,
            },
        ),
        (
            AggregationLevel.mean_per_task_type,
            {
                "mteb/baseline-random-encoder": 0.007336,
                "sentence-transformers/all-MiniLM-L6-v2": 0.645581,
            },
        ),
    ],
    ids=[AggregationLevel.mean_per_task, AggregationLevel.mean_per_task_type],
)
def test_benchmark_score(
    mock_mteb_cache: ResultCache,
    aggregation_level: AggregationLevel,
    models_scores: dict[str, float],
):
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = Benchmark(
        name="mock_benchmark",
        tasks=mteb.get_tasks(
            [
                "NanoSCIDOCSRetrieval",
                "Banking77Classification",
                "NanoArguAnaRetrieval",
            ]
        ),
    )

    mock_results = mock_mteb_cache.load_results()
    mock_random_encoder_results = mock_mteb_cache.load_results(
        models=[mock_model_name]
    ).model_results[0]
    assert np.allclose(
        mock_benchmark.get_score(
            mock_random_encoder_results, aggregation_level=aggregation_level
        ),
        models_scores[mock_model_name],
    )
    models_real_scores = mock_benchmark.get_scores(
        mock_results, aggregation_level=aggregation_level
    )
    for model_name, model_expected_score in models_scores.items():
        assert np.allclose(models_real_scores[model_name], model_expected_score)

    mock_benchmark.tasks += mteb.get_tasks(
        [
            "BelebeleRetrieval",  # in cache missing some subsets
        ]
    )

    assert (
        mock_benchmark.get_score(
            mock_random_encoder_results, aggregation_level=aggregation_level
        )
        is None
    )
    models_real_scores = mock_benchmark.get_scores(
        mock_results, aggregation_level=aggregation_level
    )

    for model_name in models_scores:
        assert models_real_scores[model_name] is None
