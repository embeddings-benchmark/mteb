from __future__ import annotations

import pytest

from mteb import MTEB
from tests.test_benchmark.mock_models import (
    MockSentenceTransformer,
)
from tests.test_benchmark.mock_tasks import (
    MockMultilingualRetrievalTask,
    MockRetrievalTask,
)


@pytest.fixture
def model():
    return MockSentenceTransformer()


@pytest.fixture
def tasks():
    return [MockRetrievalTask()]


@pytest.fixture
def multilingual_tasks():
    return [MockMultilingualRetrievalTask()]


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


def test_all_languages_evaluated(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "all_lang_evaluated"),
        verbosity=2,
        eval_langs=None,
    )
    assert "MockMultilingualRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_missing_language(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_langs=["eng"],
    )

    assert "MockMultilingualRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]

    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_langs=["eng", "fra"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_no_missing_languages(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "no_missing_lang_test"),
        verbosity=2,
        eval_langs=["eng", "fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1

    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "no_missing_lang_test"),
        verbosity=2,
        eval_langs=["eng", "fra"],
        overwrite_results=True,
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 0


def test_partial_languages(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_langs=["fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]

    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_langs=["fra", "eng"],
        overwrite_results=True,
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_multilingual_multiple_splits_partial_langs_partial_splits(
    model, multilingual_tasks, tmp_path
):
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        eval_langs=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}

    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        eval_langs=["eng", "fra"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}


def test_multilingual_multiple_splits_missing_only_one_language_in_one_split(
    model, multilingual_tasks, tmp_path
):
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_langs=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}

    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_langs=["eng"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}

    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_langs=["eng", "fra"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}
