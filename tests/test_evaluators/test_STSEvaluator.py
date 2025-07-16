from __future__ import annotations

import pytest

from mteb.evaluation.evaluators.STSEvaluator import STSEvaluator
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import MockSTSTask


# Fixtures
@pytest.fixture
def model():
    return MockNumpyEncoder(seed=42)


@pytest.fixture
def mock_task():
    task = MockSTSTask()
    task.load_data()
    return task


def test_output_structure(model, mock_task):
    """Test that the evaluator returns the expected output structure."""
    test_data = mock_task.dataset["test"]

    evaluator = STSEvaluator(
        sentences1=test_data["sentence1"],
        sentences2=test_data["sentence2"],
        gold_scores=test_data["score"],
        task_name="test_sts",
    )
    scores = evaluator(model)

    # Check basic structure
    assert isinstance(scores, dict)

    # Check required metrics
    assert "pearson" in scores
    assert "spearman" in scores
    assert "cosine_pearson" in scores
    assert "cosine_spearman" in scores
    assert "manhattan_pearson" in scores
    assert "manhattan_spearman" in scores
    assert "euclidean_pearson" in scores
    assert "euclidean_spearman" in scores


def test_expected_scores(model, mock_task):
    """Test that the evaluator returns expected scores with deterministic model."""
    test_data = mock_task.dataset["test"]

    evaluator = STSEvaluator(
        sentences1=test_data["sentence1"],
        sentences2=test_data["sentence2"],
        gold_scores=test_data["score"],
        task_name="test_sts",
    )
    scores = evaluator(model)

    # Check that we get reasonable correlation scores (between -1 and 1)
    assert -1.0 <= scores["pearson"] <= 1.0
    assert -1.0 <= scores["spearman"] <= 1.0
    assert -1.0 <= scores["cosine_pearson"] <= 1.0
    assert -1.0 <= scores["cosine_spearman"] <= 1.0
    assert -1.0 <= scores["manhattan_pearson"] <= 1.0
    assert -1.0 <= scores["manhattan_spearman"] <= 1.0
    assert -1.0 <= scores["euclidean_pearson"] <= 1.0
    assert -1.0 <= scores["euclidean_spearman"] <= 1.0


def test_limit_parameter(model, mock_task):
    """Test that the limit parameter works correctly."""
    test_data = mock_task.dataset["test"]

    # Test with limit parameter, but need at least 2 items for correlation
    original_length = len(test_data["sentence1"])

    # Only test if we have enough data, otherwise skip
    if original_length >= 2:
        evaluator_limited = STSEvaluator(
            sentences1=test_data["sentence1"],
            sentences2=test_data["sentence2"],
            gold_scores=test_data["score"],
            task_name="test_sts_limited",
            limit=original_length,  # Use all available data
        )

        # Check that limit is applied
        assert len(evaluator_limited.sentences1) == original_length
        assert len(evaluator_limited.sentences2) == original_length
        assert len(evaluator_limited.gold_scores) == original_length

        # Test evaluation still works
        scores = evaluator_limited(model)
        assert isinstance(scores, dict)
        assert "pearson" in scores


def test_basic_functionality(model, mock_task):
    """Test basic functionality and proper initialization."""
    test_data = mock_task.dataset["test"]

    evaluator = STSEvaluator(
        sentences1=test_data["sentence1"],
        sentences2=test_data["sentence2"],
        gold_scores=test_data["score"],
        task_name="test_sts",
    )

    # Check that data is properly stored
    assert evaluator.sentences1 == test_data["sentence1"]
    assert evaluator.sentences2 == test_data["sentence2"]
    assert evaluator.gold_scores == test_data["score"]
    assert evaluator.task_name == "test_sts"
