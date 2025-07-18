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
    """Test that the evaluator returns the expected output structure and scores."""
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

    # Check exact score values with deterministic model
    assert scores["pearson"] == -1.0
    assert scores["spearman"] == -0.9999999999999999
    assert scores["cosine_pearson"] == -1.0
    assert scores["cosine_spearman"] == -0.9999999999999999
    assert scores["manhattan_pearson"] == -1.0
    assert scores["manhattan_spearman"] == -0.9999999999999999
    assert scores["euclidean_pearson"] == -1.0
    assert scores["euclidean_spearman"] == -0.9999999999999999


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

    evaluator = STSEvaluator(
        sentences1=test_data["sentence1"],
        sentences2=test_data["sentence2"],
        gold_scores=test_data["score"],
        task_name="test_sts",
    )

    # Test with batch_size=1 - should process texts one at a time
    tracking_model_1 = BatchTrackingMockEncoder()
    scores_batch_1 = evaluator(tracking_model_1, encode_kwargs={"batch_size": 1})

    # Calculate expected batch calls
    num_sentences1 = len(test_data["sentence1"])
    num_sentences2 = len(test_data["sentence2"])

    # With batch_size=1, each text should be processed in its own batch
    expected_batch_1_calls = num_sentences1 + num_sentences2
    assert len(tracking_model_1.batch_calls) == expected_batch_1_calls
    assert all(batch_size == 1 for batch_size in tracking_model_1.batch_calls)

    # Test with batch_size=2 - should process texts in pairs (mostly)
    tracking_model_2 = BatchTrackingMockEncoder()
    scores_batch_2 = evaluator(tracking_model_2, encode_kwargs={"batch_size": 2})

    # Calculate expected batch calls for batch_size=2
    import math

    expected_batch_2_calls = math.ceil(num_sentences1 / 2) + math.ceil(
        num_sentences2 / 2
    )
    assert len(tracking_model_2.batch_calls) == expected_batch_2_calls
    # Each batch should have at most 2 items (last batch might have 1)
    assert all(batch_size <= 2 for batch_size in tracking_model_2.batch_calls)
    assert all(batch_size >= 1 for batch_size in tracking_model_2.batch_calls)

    # Check that evaluation works with custom batch sizes
    assert isinstance(scores_batch_1, dict)
    assert isinstance(scores_batch_2, dict)
    assert "pearson" in scores_batch_1
    assert "pearson" in scores_batch_2
