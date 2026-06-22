import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mteb._evaluators import SklearnEvaluator
from mteb.timing import TimingStack
from tests.mock_tasks import MockClassificationTask


@pytest.fixture
def mock_task():
    task = MockClassificationTask()
    task.load_data()
    return task


def test_expected_scores(mock_task):
    """Test that the evaluator returns expected scores with pre-computed embeddings."""
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    evaluator = SklearnEvaluator(
        train_data,
        test_data,
        label_column_name=mock_task.label_column_name,
        evaluator_model=LogisticRegression(max_iter=10),
        timer=TimingStack(),
    )

    rng = np.random.RandomState(42)
    train_embeddings = rng.randn(len(train_data), 8)
    test_embeddings = rng.randn(len(test_data), 8)

    y_pred = evaluator(train_embeddings, test_embeddings)

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(test_data)


def test_deterministic_predictions(mock_task):
    """Test that the evaluator produces identical results when called with the same embeddings."""
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    evaluator = SklearnEvaluator(
        train_data,
        test_data,
        label_column_name=mock_task.label_column_name,
        evaluator_model=LogisticRegression(max_iter=10),
        timer=TimingStack(),
    )

    rng = np.random.RandomState(42)
    train_embeddings = rng.randn(len(train_data), 8)
    test_embeddings = rng.randn(len(test_data), 8)

    y_pred_1 = evaluator(train_embeddings, test_embeddings)
    y_pred_2 = evaluator(train_embeddings, test_embeddings)

    assert np.array_equal(y_pred_1, y_pred_2)
