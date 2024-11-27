from __future__ import annotations

import pytest
from tests.test_benchmark.mock_models import (
    MockSentenceTransformer,
)

from tests.test_benchmark.mock_tasks import (
    MockRetrievalTask,
)

from mteb import MTEB


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
    print(results)
    assert "MockRetrievalTask" == results[0].task_name
    scores = results[0].scores
    assert set(scores.keys()) == {"val", "test"}
    assert len(scores.keys()) == 2


def test_one_missing_split(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    results1 = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    evaluation_2 = MTEB(tasks=tasks)
    results2 = evaluation_2.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation_2.get_last_evaluated_splits()

    print(last_evaluated_splits)
    assert "MockRetrievalTask" in last_evaluated_splits
    assert set(last_evaluated_splits["NFCorpus"]) == {"test"}
    assert len(last_evaluated_splits["NFCorpus"]) == 1


def test_no_missing_splits(model, tasks, tmp_path):
    evaluation_1 = MTEB(tasks=tasks)
    results = evaluation_1.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    evaluation_2 = MTEB(tasks=tasks)
    results2 = evaluation_2.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation_2.get_last_evaluated_splits()

    assert "NFCorpus" in last_evaluated_splits
    assert len(last_evaluated_splits["NFCorpus"]) == 0
