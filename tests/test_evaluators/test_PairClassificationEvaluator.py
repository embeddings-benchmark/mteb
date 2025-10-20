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
            task.dataset["test"],
            "sentence1",
            "sentence2",
            MockClassificationTask.metadata,
            "test",
            "test",
        )
        distances = evaluator(
            mteb.get_model("baseline/random-encoder-baseline"),
            encode_kwargs={"batch_size": 32},
        )
        assert distances["cosine_scores"] == pytest.approx(
            [0.7375020980834961, 0.7731508016586304], TOL
        )
        assert distances["euclidean_distances"] == pytest.approx(
            [2.4108424186706543, 2.1905980110168457], TOL
        )
        assert distances["manhattan_distances"] == pytest.approx(
            [11.177837371826172, 10.721406936645508], TOL
        )
        assert distances["similarity_scores"] == pytest.approx(
            [0.7375020384788513, 0.7731509208679199], TOL
        )
        assert distances["dot_scores"] == pytest.approx(
            [7.974165916442871, 8.176445960998535], TOL
        )
