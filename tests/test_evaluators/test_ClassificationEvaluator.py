from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mteb.evaluation.evaluators import logRegClassificationEvaluator
from tests.test_benchmark.mock_models import MockNumpyEncoder


@dataclass
class ClassificationTestCase:
    x_train: list[str]
    y_train: list[int]
    x_test: list[str]
    y_test: list[int]
    expected_score: float | None = None  # For deterministic tests


BINARY_TEST_CASE = ClassificationTestCase(
    x_train=["pos1", "pos2", "neg1", "neg2"],
    y_train=[1, 1, 0, 0],
    x_test=["new pos", "new neg"],
    y_test=[1, 0],
    expected_score=0.5,
)

MULTICLASS_TEST_CASE = ClassificationTestCase(
    x_train=["cls 1", "still cls 1", "cls 2", "also cls 2", "cls 3", "cls 3 too"],
    y_train=[0, 0, 1, 1, 2, 2],
    x_test=["new cls 1", "new cls 2", "new cls 3"],
    y_test=[0, 1, 2],
    expected_score=0.0,
)


def is_binary_classification(y_train: list[int], y_test: list[int]) -> bool:
    """Check if the classification task is binary based on the labels."""
    all_labels = set(y_train + y_test)
    return len(all_labels) == 2


# Fixtures
@pytest.fixture
def model():
    return MockNumpyEncoder(seed=42)


@pytest.fixture(params=[BINARY_TEST_CASE, MULTICLASS_TEST_CASE])
def test_case(request):
    return request.param


@pytest.fixture("test_case", [BINARY_TEST_CASE, MULTICLASS_TEST_CASE])
def test_output_structure(model, test_case: ClassificationTestCase):
    """Test that the evaluator returns the expected output structure."""
    evaluator = logRegClassificationEvaluator(
        test_case.x_train,
        np.array(test_case.y_train),
        test_case.x_test,
        np.array(test_case.y_test),
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

    # Check binary-specific metrics
    is_binary = is_binary_classification(test_case.y_train, test_case.y_test)
    if is_binary:
        assert "ap" in scores
        assert "ap_weighted" in scores
    else:
        assert "ap" not in scores


@pytest.fixture("test_case", [BINARY_TEST_CASE, MULTICLASS_TEST_CASE])
def test_expected_scores(model, test_case: ClassificationTestCase):
    """Test that the evaluator returns expected scores with deterministic model."""
    if test_case.expected_score is None:
        pytest.skip("No expected score defined for this test case")

    evaluator = logRegClassificationEvaluator(
        test_case.x_train,
        np.array(test_case.y_train),
        test_case.x_test,
        np.array(test_case.y_test),
        task_name="test_classification",
    )
    scores, _ = evaluator(model)

    # Check that accuracy matches expected value (with some tolerance for floating point)
    assert abs(scores["accuracy"] - test_case.expected_score) < 1e-10, (
        f"Expected accuracy {test_case.expected_score}, got {scores['accuracy']}"
    )


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
    test_case = BINARY_TEST_CASE
    model = MockNumpyEncoder(seed=42)

    # First evaluation to generate cache
    evaluator_initial = logRegClassificationEvaluator(
        test_case.x_train,
        np.array(test_case.y_train),
        test_case.x_test,
        np.array(test_case.y_test),
        task_name="test_binary_cache",
    )
    _, test_cache_initial = evaluator_initial(model)

    # Second evaluation using cache
    evaluator_with_cache = logRegClassificationEvaluator(
        test_case.x_train,
        np.array(test_case.y_train),
        test_case.x_test,
        np.array(test_case.y_test),
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
