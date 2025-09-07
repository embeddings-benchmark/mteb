from __future__ import annotations

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
    assert scores["pearson"] == 1.0
    assert scores["spearman"] == 0.9999999999999999
    assert scores["cosine_pearson"] == 1.0
    assert scores["cosine_spearman"] == 0.9999999999999999
    assert scores["manhattan_pearson"] == 1.0
    assert scores["manhattan_spearman"] == 0.9999999999999999
    assert scores["euclidean_pearson"] == 1.0
    assert scores["euclidean_spearman"] == 0.9999999999999999
