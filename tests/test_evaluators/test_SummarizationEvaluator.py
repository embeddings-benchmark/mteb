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
    """Test that the evaluator returns the expected output structure and scores."""
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

    # Check exact score values with deterministic model
    assert scores["pearson"] == -1.0
    assert scores["spearman"] == -0.9999999999999999
    assert scores["cosine_pearson"] == -1.0
    assert scores["cosine_spearman"] == -0.9999999999999999
    assert scores["dot_pearson"] == 0.0
    assert scores["dot_spearman"] == 0.0


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


def test_batch_size_parameter(mock_task):
    """Test that the batch_size parameter in encode_kwargs works correctly."""
    from tests.test_benchmark.mock_models import MockNumpyEncoder

    # Create a mock encoder that respects batch_size and tracks batch calls
    class BatchTrackingMockEncoder(MockNumpyEncoder):
        def __init__(self, seed=42):
            super().__init__(seed)
            self.batch_calls = []  # Track each batch call

        def encode(self, sentences, prompt_name=None, **kwargs):
            batch_size = kwargs.get("batch_size", 32)

            # Track individual batch calls
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                self.batch_calls.append(len(batch))

            return super().encode(sentences, prompt_name, **kwargs)

    test_data = mock_task.dataset["test"]

    evaluator = SummarizationEvaluator(
        human_summaries=test_data["human_summaries"],
        machine_summaries=test_data["machine_summaries"],
        gold_scores=test_data["relevance"],
        task_name="test_summarization",
    )

    # Test with batch_size=1 - should process texts one at a time
    tracking_model_1 = BatchTrackingMockEncoder()
    scores_batch_1 = evaluator(tracking_model_1, encode_kwargs={"batch_size": 1})

    # Calculate expected batch calls
    total_human_texts = sum(len(hs) for hs in test_data["human_summaries"])
    total_machine_texts = sum(len(ms) for ms in test_data["machine_summaries"])

    # With batch_size=1, each text should be processed in its own batch
    expected_batch_1_calls = total_human_texts + total_machine_texts
    assert len(tracking_model_1.batch_calls) == expected_batch_1_calls
    assert all(batch_size == 1 for batch_size in tracking_model_1.batch_calls)

    # Test with batch_size=2 - should process texts in pairs (mostly)
    tracking_model_2 = BatchTrackingMockEncoder()
    scores_batch_2 = evaluator(tracking_model_2, encode_kwargs={"batch_size": 2})

    # Calculate expected batch calls for batch_size=2
    import math

    expected_batch_2_calls = math.ceil(total_human_texts / 2) + math.ceil(
        total_machine_texts / 2
    )
    assert len(tracking_model_2.batch_calls) == expected_batch_2_calls
    # Each batch should have at most 2 items (last batch might have 1)
    assert all(1 <= batch_size <= 2 for batch_size in tracking_model_2.batch_calls)

    # Check that evaluation works with custom batch sizes
    assert isinstance(scores_batch_1, dict)
    assert isinstance(scores_batch_2, dict)
    assert "pearson" in scores_batch_1
    assert "pearson" in scores_batch_2


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
