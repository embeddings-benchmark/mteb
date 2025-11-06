import pytest

import mteb
from mteb._evaluators import SummarizationEvaluator
from tests.mock_tasks import MockSummarizationTask


# Fixtures
@pytest.fixture
def model():
    return mteb.get_model("baseline/random-encoder-baseline")


@pytest.fixture
def mock_task():
    task = MockSummarizationTask()
    task.load_data()
    return task


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
