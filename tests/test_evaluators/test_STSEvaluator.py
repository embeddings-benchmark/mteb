import pytest

from mteb._evaluators import AnySTSEvaluator
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

    evaluator = AnySTSEvaluator(
        test_data,
        mock_task.column_names,
        gold_scores=list(map(mock_task.normalize, test_data["score"])),
        task_metadata=mock_task.metadata,
        hf_subset="default",
        hf_split="test",
    )
    scores = evaluator(model, encode_kwargs={"batch_size": 32})

    # Check basic structure
    assert isinstance(scores, dict)

    assert "cosine_scores" in scores
    assert "manhattan_distances" in scores
    assert "euclidean_distances" in scores
    assert "similarity_scores" in scores
