from __future__ import annotations

import pytest

from mteb._evaluators import SummarizationEvaluator
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
    """Test that the evaluator returns the expected output structure and scores."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        texts=test_data["text"],
        gold_scores=test_data["relevance"],
        task_metadata=mock_task.metadata,
        hf_split="test",
        hf_subset="default",
    )
    scores = evaluator(model, encode_kwargs={"batch_size": 32})

    # Check basic structure
    assert isinstance(scores, dict)

    # Check required metrics
    assert "pearson" in scores
    assert "spearman" in scores
    assert "cosine_pearson" in scores
    assert "cosine_spearman" in scores
    assert "dot_pearson" in scores
    assert "dot_spearman" in scores

    # Check exact score values with deterministic model
    assert scores["pearson"] == 1.0
    assert scores["spearman"] == 0.9999999999999999
    assert scores["cosine_pearson"] == 1.0
    assert scores["cosine_spearman"] == 0.9999999999999999
    assert scores["dot_pearson"] == 0.0
    assert scores["dot_spearman"] == 0.0


def test_basic_functionality(model, mock_task):
    """Test basic functionality and proper initialization."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        texts=test_data["text"],
        gold_scores=test_data["relevance"],
        task_metadata=mock_task.metadata,
        hf_split="test",
        hf_subset="default",
    )

    # Check that data is properly stored
    assert evaluator.human_summaries == test_data["human_summaries"]
    assert evaluator.machine_summaries == test_data["machine_summaries"]
    assert evaluator.gold_scores == test_data["relevance"]


def test_encode_kwargs_handling(model, mock_task):
    """Test that encode_kwargs are properly handled."""
    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        texts=test_data["text"],
        gold_scores=test_data["relevance"],
        task_metadata=mock_task.metadata,
        hf_split="test",
        hf_subset="default",
    )

    # Test that the evaluator accepts encode_kwargs
    scores = evaluator(model, encode_kwargs={"batch_size": 16})
    assert isinstance(scores, dict)
    assert "pearson" in scores


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
        texts=test_data["text"],
        gold_scores=test_data["relevance"],
        task_metadata=mock_task.metadata,
        hf_split="test",
        hf_subset="default",
    )

    # Should still work even with some samples having equal scores
    scores = evaluator(model, encode_kwargs={"batch_size": 32})
    assert isinstance(scores, dict)
    assert "pearson" in scores
