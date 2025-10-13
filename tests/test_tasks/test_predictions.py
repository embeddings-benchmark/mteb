import json
from pathlib import Path

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
    MockTextZeroShotClassificationTask,
)


@pytest.mark.parametrize(
    "task_cls, expected",
    [
        (
            MockBitextMiningTask,
            {
                "sentence1-sentence2": [
                    {"corpus_id": 1, "score": 0.7544079055482198},
                    {"corpus_id": 1, "score": 0.7991580438841115},
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
                [0.9999999999381253, 0.7463670084501418],
                [0.7463670084501418, 0.9999999999291757],
            ],
        ),
        (
            MockRetrievalTask,
            {
                "q1": {"d1": pytest.approx(0.7990461275020804)},
                "q2": {"d1": pytest.approx(0.7959300710973475)},
            },
        ),
        (
            MockSTSTask,
            {
                "cosine_scores": [
                    pytest.approx(0.7420341674650179),
                    pytest.approx(0.799158043937249),
                ],
                "euclidean_distances": [
                    pytest.approx(-2.368116011174239),
                    pytest.approx(-1.9505613193234193),
                ],
                "manhattan_distances": [
                    pytest.approx(-11.10350771249041),
                    pytest.approx(-8.983517484440046),
                ],
                "similarity_scores": [
                    pytest.approx(0.742034167419937),
                    pytest.approx(0.7991580438841115),
                ],
            },
        ),
        (
            MockPairClassificationTask,
            {
                "cosine_scores": [
                    pytest.approx(0.7420341674650179),
                    pytest.approx(0.799158043937249),
                ],
                "dot_scores": [
                    pytest.approx(8.044359667700071),
                    pytest.approx(7.261175109797408),
                ],
                "euclidean_distances": [
                    pytest.approx(2.368116011174239),
                    pytest.approx(1.9505613193234193),
                ],
                "manhattan_distances": [
                    pytest.approx(11.10350771249041),
                    pytest.approx(8.983517484440046),
                ],
                "similarity_scores": [
                    pytest.approx(0.742034167419937),
                    pytest.approx(0.7991580438841115),
                ],
            },
        ),
        (MockClusteringTask, [[2, 1, 0]]),
        (MockClusteringFastTask, {"Level 0": [[1, 2, 2], [0, 1, 0], [0, 1, 1]]}),
        (
            MockRegressionTask,
            [[0.9999999999999999, 0.5297196305892906]] * 10,
        ),
        (
            MockImageTextPairClassificationTask,
            [
                [[pytest.approx(0.8513203857500926)]],
                [[pytest.approx(0.7686800183685046)]],
            ],
        ),
    ],
)
def test_predictions(tmp_path: Path, task_cls, expected):
    """Run evaluation for each mock task and check predictions."""
    task = task_cls()
    model = mteb.get_model_meta("mteb/random-encoder-baseline")
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        full_predictions = json.load(f)

    predictions = full_predictions["default"]["test"]
    assert predictions == expected
