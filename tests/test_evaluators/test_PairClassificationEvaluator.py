import pytest

from mteb._evaluators import PairClassificationEvaluator
from tests.mock_models import MockNumpyEncoder
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
        distances = evaluator(MockNumpyEncoder(), encode_kwargs={"batch_size": 32})
        assert distances["cosine_scores"] == pytest.approx(
            [0.6676752688013152, 0.44889138491964875], TOL
        )
        assert distances["euclidean_distances"] == pytest.approx(
            [1.4245895965840156, 1.8406867696540525], TOL
        )
        assert distances["manhattan_distances"] == pytest.approx(
            [3.8137664259525317, 5.127750669852312], TOL
        )
        assert distances["similarity_scores"] == pytest.approx(
            [0.6676753163337708, 0.44889140129089355], TOL
        )
        assert distances["dot_scores"] == pytest.approx(
            [1.834609502285649, 1.3250427203346034], TOL
        )
