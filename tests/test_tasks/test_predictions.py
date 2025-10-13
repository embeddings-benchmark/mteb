import json
from pathlib import Path
from typing import Any

import pytest

import mteb
from tests.mock_tasks import (
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


@pytest.mark.parametrize(
    "task_cls, expected",
    [
        (
            MockBitextMiningTask,
            {
                "sentence1-sentence2": [
                    {"corpus_id": 1, "score": 0.7652846501709707},
                    {"corpus_id": 1, "score": 0.7652846501709707},
                ]
            },
        ),
        (
            MockClassificationTask,
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ),
        (
            MockTextZeroShotClassificationTask,
            [
                [0.9999999999419484, 0.9999999999419484],
                [0.9999999999419484, 0.9999999999419484],
            ],
        ),
        (
            MockRetrievalTask,
            {"q1": {"d2": 0.9999999999419484}, "q2": {"d2": 0.9999999999419484}},
        ),
        (
            MockSTSTask,
            {
                "cosine_scores": [
                    pytest.approx(0.7271937024333304),
                    pytest.approx(0.7652846502146503),
                ],
                "manhattan_distances": [
                    pytest.approx(-11.606113294587654),
                    pytest.approx(-11.687107717574829),
                ],
                "euclidean_distances": [
                    pytest.approx(-2.4683132356991315),
                    pytest.approx(-2.404168159957386),
                ],
                "similarity_scores": [
                    pytest.approx(0.7271937023895851),
                    pytest.approx(0.7652846501709707),
                ],
            },
        ),
        (
            MockPairClassificationTask,
            {
                "cosine_scores": [
                    pytest.approx(0.7271937024333304),
                    pytest.approx(0.7652846502146503),
                ],
                "euclidean_distances": [
                    pytest.approx(2.4683132356991315),
                    pytest.approx(2.404168159957386),
                ],
                "manhattan_distances": [
                    pytest.approx(11.606113294587654),
                    pytest.approx(11.687107717574829),
                ],
                "similarity_scores": [
                    pytest.approx(0.7271937023895851),
                    pytest.approx(0.7652846501709707),
                ],
                "dot_scores": [
                    pytest.approx(8.047867836468026),
                    pytest.approx(9.399434014739104),
                ],
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
    model = mteb.get_model_meta("mteb/random-baseline")
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        full_predictions = json.load(f)

    predictions = full_predictions["default"]["test"]
    assert predictions == expected
