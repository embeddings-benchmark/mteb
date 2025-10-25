import json
from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.models import SearchEncoderWrapper
from mteb.models.search_encoder_index import (
    DefaultEncoderSearchBackend,
    FaissEncoderSearchBackend,
)
from tests.mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringTask,
    MockImageTextPairClassificationTask,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
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
                "q1": {"d1": pytest.approx(0.7787374258041382)},
                "q2": {"d1": pytest.approx(0.7900890707969666)},
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


@pytest.mark.parametrize(
    "task",
    [
        MockRetrievalTask(),
        MockRerankingTask(),
    ],
)
def test_retrieval_backends(task: AbsTaskRetrieval, tmp_path: Path):
    """Test different retrieval backends for retrieval and reranking tasks."""
    model = mteb.get_model("baseline/random-encoder-baseline")

    python_backend = SearchEncoderWrapper(
        model, index_backend=DefaultEncoderSearchBackend()
    )
    faiss_backend = SearchEncoderWrapper(
        model, index_backend=FaissEncoderSearchBackend(model)
    )

    python_backend_predictions = tmp_path / "python_backend_predictions"
    faiss_backend_predictions = tmp_path / "faiss_backend_predictions"

    python_results = mteb.evaluate(
        python_backend,
        task,
        prediction_folder=python_backend_predictions,
        cache=None,
    )
    faiss_results = mteb.evaluate(
        faiss_backend,
        task,
        prediction_folder=faiss_backend_predictions,
        cache=None,
    )

    assert (
        python_results.task_results[0].get_score()
        == faiss_results.task_results[0].get_score()
    )

    with task._predictions_path(python_backend_predictions).open() as f:
        full_python_predictions = json.load(f)
        python_predictions = full_python_predictions["default"]["test"]

    with task._predictions_path(faiss_backend_predictions).open() as f:
        full_faiss_predictions = json.load(f)
        faiss_predictions = full_faiss_predictions["default"]["test"]

    for python_pred_key, faiss_pred_key in zip(
        sorted(python_predictions.keys()), sorted(faiss_predictions.keys())
    ):
        assert python_pred_key == faiss_pred_key
        assert python_predictions[python_pred_key] == pytest.approx(
            faiss_predictions[faiss_pred_key]
        )
