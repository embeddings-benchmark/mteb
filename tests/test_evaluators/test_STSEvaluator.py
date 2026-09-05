import pytest

import mteb
from mteb._evaluators import AnySTSEvaluator
from mteb.mocks.mock_tasks import MockSTSTask
from mteb.models.models_protocols import EncoderProtocol
from mteb.timing import TimingStack


# Fixtures
@pytest.fixture
def model() -> EncoderProtocol:
    return mteb.get_model("mteb/baseline-random-encoder")


@pytest.fixture
def mock_task() -> MockSTSTask:
    task = MockSTSTask()
    task.load_data()
    return task


def test_output_structure(model: EncoderProtocol, mock_task: MockSTSTask) -> None:
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
        timer=TimingStack(),
    )
    scores = evaluator(model, encode_kwargs={"batch_size": 32})

    # Check basic structure
    assert isinstance(scores, dict)

    assert "cosine_scores" in scores
    assert "manhattan_distances" in scores
    assert "euclidean_distances" in scores
    assert "similarity_scores" in scores
