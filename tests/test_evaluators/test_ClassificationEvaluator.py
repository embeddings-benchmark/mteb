import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import mteb
from mteb._evaluators import SklearnEvaluator
from tests.mock_tasks import MockClassificationTask


# Fixtures
@pytest.fixture
def model():
    return mteb.get_model("baseline/random-encoder-baseline")


@pytest.fixture
def mock_task():
    task = MockClassificationTask()
    task.load_data()
    return task


def test_expected_scores(model, mock_task):
    """Test that the evaluator returns expected scores with deterministic model."""
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    evaluator = SklearnEvaluator(
        train_data,
        test_data,
        mock_task.input_column_name,
        mock_task.label_column_name,
        mock_task.metadata,
        hf_split="test",
        hf_subset="default",
        evaluator_model=LogisticRegression(
            n_jobs=-1,
            max_iter=10,
        ),
    )
    y_pred, test_cache = evaluator(model, encode_kwargs={"batch_size": 32})

    assert isinstance(y_pred, np.ndarray)
    assert isinstance(test_cache, np.ndarray)

    # Check that we get reasonable scores (MockClassificationTask has deterministic data)
    assert y_pred.tolist() == [0, 1]


def test_cache_usage_binary(model):
    """Test that embedding caching works correctly for binary classification.

    This test verifies the caching mechanism used to avoid re-encoding the same
    sentences multiple times. The workflow is:

    1. Run a first evaluation which encodes test sentences and returns embeddings cache
    2. Run a second evaluation with the same test sentences, passing the cache from step 1
    3. Verify that the cache is preserved (not modified) during the second evaluation
    4. Verify that the second evaluation still produces valid classification results

    The cache contains the encoded embeddings for the test sentences, allowing the
    evaluator to skip the encoding step when the same sentences are evaluated again.
    This is particularly useful when running multiple evaluations on the same dataset
    with different models or parameters.
    """
    mock_task = MockClassificationTask()
    mock_task.load_data()
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    # First evaluation to generate cache
    evaluator_initial = SklearnEvaluator(
        train_data,
        test_data,
        mock_task.input_column_name,
        mock_task.label_column_name,
        mock_task.metadata,
        hf_split="test",
        hf_subset="default",
        evaluator_model=LogisticRegression(
            n_jobs=-1,
            max_iter=10,
        ),
    )
    _, test_cache_initial = evaluator_initial(model, encode_kwargs={"batch_size": 32})

    # Second evaluation using cache
    evaluator_with_cache = SklearnEvaluator(
        train_data,
        test_data,
        mock_task.input_column_name,
        mock_task.label_column_name,
        mock_task.metadata,
        hf_split="test",
        hf_subset="default",
        evaluator_model=LogisticRegression(
            n_jobs=-1,
            max_iter=10,
        ),
    )
    y_pred, test_cache_after_cache_usage = evaluator_with_cache(
        model, encode_kwargs={"batch_size": 32}, test_cache=test_cache_initial
    )

    # Verify cache is preserved
    assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)
    assert y_pred.tolist() == [0, 1]
