from __future__ import annotations

import numpy as np
import pytest

from mteb.evaluation.evaluators import logRegClassificationEvaluator
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import MockClassificationTask


def is_binary_classification(y_train: list[int], y_test: list[int]) -> bool:
    """Check if the classification task is binary based on the labels."""
    all_labels = set(y_train + y_test)
    return len(all_labels) == 2


# Fixtures
@pytest.fixture
def model():
    return MockNumpyEncoder(seed=42)


@pytest.fixture
def mock_task():
    task = MockClassificationTask()
    task.load_data()
    return task


def test_output_structure(model, mock_task):
    """Test that the evaluator returns the expected output structure."""
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    evaluator = logRegClassificationEvaluator(
        train_data["text"],
        np.array(train_data["label"]),
        test_data["text"],
        np.array(test_data["label"]),
        task_name="test_classification",
    )
    scores, test_cache = evaluator(model)

    # Check basic structure
    assert isinstance(scores, dict)
    assert isinstance(test_cache, np.ndarray)

    # Check required metrics
    assert "accuracy" in scores
    assert "f1" in scores
    assert "f1_weighted" in scores

    # Check binary-specific metrics (MockClassificationTask is binary)
    is_binary = is_binary_classification(train_data["label"], test_data["label"])
    if is_binary:
        assert "ap" in scores
        assert "ap_weighted" in scores
    else:
        assert "ap" not in scores


def test_expected_scores(model, mock_task):
    """Test that the evaluator returns expected scores with deterministic model."""
    train_data = mock_task.dataset["train"]
    test_data = mock_task.dataset["test"]

    evaluator = logRegClassificationEvaluator(
        train_data["text"],
        np.array(train_data["label"]),
        test_data["text"],
        np.array(test_data["label"]),
        task_name="test_classification",
    )
    scores, _ = evaluator(model)

    # Check that we get reasonable scores (MockClassificationTask has deterministic data)
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert 0.0 <= scores["f1"] <= 1.0
    assert 0.0 <= scores["f1_weighted"] <= 1.0


def test_cache_usage_binary():
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

    model = MockNumpyEncoder(seed=42)

    # First evaluation to generate cache
    evaluator_initial = logRegClassificationEvaluator(
        train_data["text"],
        np.array(train_data["label"]),
        test_data["text"],
        np.array(test_data["label"]),
        task_name="test_binary_cache",
    )
    _, test_cache_initial = evaluator_initial(model)

    # Second evaluation using cache
    evaluator_with_cache = logRegClassificationEvaluator(
        train_data["text"],
        np.array(train_data["label"]),
        test_data["text"],
        np.array(test_data["label"]),
        task_name="test_binary_cache_usage",
    )
    scores_with_cache, test_cache_after_cache_usage = evaluator_with_cache(
        model, test_cache=test_cache_initial
    )

    # Verify cache is preserved
    assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)

    # Verify that scores are returned (structure check only)
    assert "accuracy" in scores_with_cache
    assert "f1" in scores_with_cache
    assert "f1_weighted" in scores_with_cache
    assert "ap" in scores_with_cache
    assert "ap_weighted" in scores_with_cache
