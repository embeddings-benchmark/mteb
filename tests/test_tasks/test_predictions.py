import json
from pathlib import Path
from typing import Any

import pytest

import mteb
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringFastTask,
    MockClusteringTask,
    MockImageTextPairClassificationTask,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockTextZeroShotClassificationTask,
)


def _round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round all float values inside nested dicts/lists."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, list):
        return [_round_floats(i, ndigits) for i in obj]
    elif isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


@pytest.mark.parametrize(
    "task_cls, expected",
    [
        (
            MockBitextMiningTask,
            {
                "sentence1-sentence2": [
                    {"corpus_id": 0, "score": 0.67},
                    {"corpus_id": 0, "score": 0.58},
                ]
            },
        ),
        (
            MockClassificationTask,
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
        ),
        (
            MockTextZeroShotClassificationTask,
            [[0.67, 0.58], [0.58, 0.45]],
        ),
        (
            MockRetrievalTask,
            {"q1": {"d2": 0.67}, "q2": {"d2": 0.58}},
        ),
        (
            MockSTSTask,
            {
                "cosine_scores": [0.67, 0.45],
                "manhattan_distances": [-3.81, -5.13],
                "euclidean_distances": [-1.42, -1.84],
                "similarity_scores": [0.67, 0.45],
            },
        ),
        (
            MockPairClassificationTask,
            {
                "cosine_scores": [0.67, 0.45],
                "euclidean_distances": [1.42, 1.84],
                "manhattan_distances": [3.81, 5.13],
                "similarity_scores": [0.67, 0.45],
                "dot_scores": [1.83, 1.33],
            },
        ),
        (
            MockSummarizationTask,
            {
                "cosine_scores": [[0.75, 0.74], [0.69, 0.92]],
                "dot_scores": [[2.16, 2.99], [1.99, 3.09]],
                "similarity_scores": [[0.75, 0.74], [0.69, 0.92]],
                "human_scores": [[1.0, 0.0], [0.0, 1.0]],
            },
        ),
        (MockClusteringTask, [[2, 2, 1]]),
        (MockClusteringFastTask, {"Level 0": [[1, 2, 2], [0, 1, 2], [2, 0, 0]]}),
        (
            MockRegressionTask,
            [[0.39, 0.76]] * 10,
        ),
        (
            MockImageTextPairClassificationTask,
            [[[0.67]], [[0.45]]],
        ),
    ],
)
def test_predictions(tmp_path: Path, task_cls, expected):
    """Run evaluation for each mock task and check predictions."""
    task = task_cls()
    model = MockNumpyEncoder()
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        full_predictions = json.load(f)

    predictions = full_predictions["default"]["test"]
    assert _round_floats(predictions, 2) == expected
