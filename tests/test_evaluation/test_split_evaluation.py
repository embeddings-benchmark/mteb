from __future__ import annotations

import pytest

from mteb import MTEB
from tests.test_benchmark.mock_models import (
    MockSentenceTransformer,
)
from tests.test_benchmark.mock_tasks import (
    MockRetrievalTask,
)


@pytest.fixture
def model():
    return MockSentenceTransformer()


@pytest.fixture
def tasks():
    return [MockRetrievalTask()]


def test_all_splits_evaluated(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "all_splits_evaluated"),
        verbosity=2,
    )

    assert "MockRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"val", "test"}
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 2


def test_one_missing_split(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    assert "MockRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"val"}
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 1

    results2 = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
        overwrite_results=True,
    )

    assert "MockRetrievalTask" == results2[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"test"}
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 1


def test_no_missing_splits(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 2

    evaluation = MTEB(tasks=tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 0
