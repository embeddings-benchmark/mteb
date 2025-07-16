from __future__ import annotations

import pytest

from mteb.evaluation.evaluators.SummarizationEvaluator import SummarizationEvaluator
from tests.test_benchmark.mock_models import MockNumpyEncoder
from tests.test_benchmark.mock_tasks import MockSummarizationTask


# Fixtures
@pytest.fixture
def model():
    return MockNumpyEncoder(seed=42)


@pytest.fixture
def mock_task():
    task = MockSummarizationTask()
    task.load_data()
    return task


def test_output_structure(model, mock_task):
    """Test that the evaluator returns the expected output structure."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )
    scores = evaluator(model)

    # Check basic structure
    assert isinstance(scores, dict)

    # Check required metrics
    assert "pearson" in scores
    assert "spearman" in scores
    assert "cosine_pearson" in scores
    assert "cosine_spearman" in scores
    assert "dot_pearson" in scores
    assert "dot_spearman" in scores


def test_expected_scores(model, mock_task):
    """Test that the evaluator returns expected scores with deterministic model."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )
    scores = evaluator(model)

    # Check that we get reasonable correlation scores (between -1 and 1)
    assert -1.0 <= scores["pearson"] <= 1.0
    assert -1.0 <= scores["spearman"] <= 1.0
    assert -1.0 <= scores["cosine_pearson"] <= 1.0
    assert -1.0 <= scores["cosine_spearman"] <= 1.0
    assert -1.0 <= scores["dot_pearson"] <= 1.0
    assert -1.0 <= scores["dot_spearman"] <= 1.0


def test_basic_functionality(model, mock_task):
    """Test basic functionality and proper initialization."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )

    # Check that data is properly stored
    assert evaluator.human_summaries == test_data["human_summaries"]
    assert evaluator.machine_summaries == test_data["machine_summaries"]
    assert evaluator.gold_scores == test_data["relevance"]
    assert evaluator.task_name == "test_summarization"


def test_encode_kwargs_handling(model, mock_task):
    """Test that encode_kwargs are properly handled."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )

    # Test that the evaluator accepts encode_kwargs
    scores = evaluator(model, encode_kwargs={"batch_size": 16})
    assert isinstance(scores, dict)
    assert "pearson" in scores


def test_batch_size_parameter(model, mock_task):
    """Test that the batch_size parameter in encode_kwargs works correctly."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )

    # Test with custom batch size
    scores = evaluator(model, encode_kwargs={"batch_size": 16})

    # Check that evaluation still works with custom batch size
    assert isinstance(scores, dict)
    assert "pearson" in scores
    assert -1.0 <= scores["pearson"] <= 1.0


def test_empty_scores_handling(model, mock_task):
    """Test that the evaluator handles cases where some samples have equal scores."""
    test_data = mock_task.dataset["test"]

    # Create a case where some gold scores are identical
    modified_gold_scores = test_data["relevance"].copy()
    if len(modified_gold_scores) > 0 and len(modified_gold_scores[0]) > 1:
        # Make all scores in the first sample identical
        modified_gold_scores[0] = [modified_gold_scores[0][0]] * len(
            modified_gold_scores[0]
        )

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=modified_gold_scores,
        task_name="test_summarization_equal_scores",
    )

    # Should still work even with some samples having equal scores
    scores = evaluator(model)
    assert isinstance(scores, dict)
    assert "pearson" in scores
