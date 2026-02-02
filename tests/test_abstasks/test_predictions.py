import json
from pathlib import Path

import pytest

import mteb
from tests.mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringTask,
    MockImageTextPairClassificationTask,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRetrievalTask,
    MockSTSTask,
    MockTextZeroShotClassificationTask,
)


@pytest.mark.parametrize(
    "task, expected",
    [
        (
            MockBitextMiningTask(),
            {
                "sentence1-sentence2": [
                    pytest.approx({"corpus_id": 0, "score": 0.7375020384788513}),
                    pytest.approx({"corpus_id": 1, "score": 0.7731509208679199}),
                ]
            },
        ),
        (
            MockClassificationTask(),
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
        ),
        (
            MockTextZeroShotClassificationTask(),
            [
                pytest.approx([0.9999998807907104, 0.6840636730194092]),
                pytest.approx([0.6840636730194092, 1.0]),
            ],
        ),
        (
            MockRetrievalTask(),
            {
                "q1": {
                    "d2": pytest.approx(0.7302242517471313),
                    "d1": pytest.approx(0.7787374258041382),
                },
                "q2": {
                    "d2": pytest.approx(0.7675943970680237),
                    "d1": pytest.approx(0.7900890707969666),
                },
            },
        ),
        (
            MockSTSTask(),
            pytest.approx(
                {
                    "cosine_scores": pytest.approx(
                        [0.7375020980834961, 0.7731508016586304]
                    ),
                    "euclidean_distances": pytest.approx(
                        [-2.4108424186706543, -2.1905980110168457]
                    ),
                    "manhattan_distances": pytest.approx(
                        [-11.177837371826172, -10.721406936645508]
                    ),
                    "similarity_scores": pytest.approx(
                        [0.7375020384788513, 0.7731509208679199]
                    ),
                }
            ),
        ),
        (
            MockPairClassificationTask(),
            {
                "cosine_scores": pytest.approx(
                    [0.7375020980834961, 0.7731508016586304]
                ),
                "dot_scores": pytest.approx([7.974165916442871, 8.176445960998535]),
                "euclidean_distances": pytest.approx(
                    [2.4108424186706543, 2.1905980110168457]
                ),
                "manhattan_distances": pytest.approx(
                    [11.177837371826172, 10.721406936645508]
                ),
                "similarity_scores": pytest.approx(
                    [0.7375020384788513, 0.7731509208679199]
                ),
            },
        ),
        (MockClusteringTask(), [[1, 2, 0]]),
        # TODO: #3441
        # Disabled due to being too flaky.
        # (
        #     MockClusteringFastTask(seed=1),
        #     {"Level 0": [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 2, 1]]},
        # ),
        (
            MockRegressionTask(),
            [pytest.approx([1.0000001192092896, 0.33665788173675537])] * 10,
        ),
        (
            MockImageTextPairClassificationTask(),
            [
                [pytest.approx([0.8081411123275757])],
                [pytest.approx([0.6950531601905823])],
            ],
        ),
    ],
)
def test_predictions(tmp_path: Path, task, expected):
    """Run evaluation for each mock task and check predictions."""
    model = mteb.get_model_meta("baseline/random-encoder-baseline")
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        full_predictions = json.load(f)

    predictions = full_predictions["default"]["test"]
    assert predictions == expected
