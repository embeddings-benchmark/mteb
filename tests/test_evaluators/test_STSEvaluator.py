import pytest

import mteb
from mteb._evaluators import AnySTSEvaluator
from tests.mock_tasks import MockSTSTask


# Fixtures
@pytest.fixture
def model():
    return mteb.get_model("baseline/random-encoder-baseline")


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
        gold_scores=list(map(mock_task._normalize, test_data["score"])),
        task_metadata=mock_task.metadata,
        hf_subset="default",
        hf_split="test",
        input1_prompt_type=None,
        input2_prompt_type=None,
    )
    scores = evaluator(model, encode_kwargs={"batch_size": 32})

    # Check basic structure
    assert isinstance(scores, dict)

    assert "cosine_scores" in scores
    assert "manhattan_distances" in scores
    assert "euclidean_distances" in scores
    assert "similarity_scores" in scores
