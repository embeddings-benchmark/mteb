import pytest

import mteb
from mteb._evaluators import PairClassificationEvaluator
from tests.mock_tasks import (
    MockClassificationTask,
    MockPairClassificationTask,
)

TOL = 0.0001


class TestPairClassificationEvaluator:
    def test_accuracy(self):
        task = MockPairClassificationTask()
        task.load_data()
        evaluator = PairClassificationEvaluator(
            task.dataset["test"]["sentence1"],
            task.dataset["test"]["sentence2"],
            MockClassificationTask.metadata,
            "test",
            "test",
        )
        distances = evaluator(
            mteb.get_model("mock/random-encoder-baseline"),
            encode_kwargs={"batch_size": 32},
        )
        assert distances["cosine_scores"] == pytest.approx(
            [0.7420341674650179, 0.799158043937249], TOL
        )
        assert distances["euclidean_distances"] == pytest.approx(
            [2.368116011174239, 1.9505613193234193], TOL
        )
        assert distances["manhattan_distances"] == pytest.approx(
            [11.10350771249041, 8.983517484440046], TOL
        )
        assert distances["similarity_scores"] == pytest.approx(
            [0.742034167419937, 0.7991580438841115], TOL
        )
        assert distances["dot_scores"] == pytest.approx(
            [8.044359667700071, 7.261175109797408], TOL
        )
