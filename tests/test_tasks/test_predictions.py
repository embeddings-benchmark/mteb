import json
from pathlib import Path

import pytest

import mteb
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockPairClassificationTask,
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
                    {"corpus_id": 0, "score": 0.667675256729126},
                    {"corpus_id": 0, "score": 0.5796383023262024},
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
            [
                [0.667675256729126, 0.5808462500572205],
                [0.5796383023262024, 0.44889137148857117],
            ],
        ),
        (
            MockRetrievalTask,
            {"q1": {"d2": 0.667675256729126}, "q2": {"d2": 0.5796383023262024}},
        ),
        (
            MockSTSTask,
            {
                "cosine_scores": [0.6676752688013152, 0.44889138491964875],
                "manhattan_distances": [-3.8137664259525317, -5.127750669852312],
                "euclidean_distances": [-1.4245895965840156, -1.8406867696540525],
                "similarity_scores": [0.6676753163337708, 0.44889140129089355],
            },
        ),
        (
            MockPairClassificationTask,
            {
                "cosine_scores": [0.6676752688013152, 0.44889138491964875],
                "euclidean_distances": [1.4245895965840156, 1.8406867696540525],
                "manhattan_distances": [3.8137664259525317, 5.127750669852312],
                "similarity_scores": [0.6676753163337708, 0.44889140129089355],
                "dot_scores": [1.834609502285649, 1.3250427203346034],
            },
        ),
    ],
)
def test_predictions(tmp_path: Path, task_cls, expected):
    """Run evaluation for each mock task and check predictions match exactly."""
    task = task_cls()
    model = MockNumpyEncoder()
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        full_predictions = json.load(f)

    predictions = full_predictions["default"]["test"]
    assert predictions == expected
