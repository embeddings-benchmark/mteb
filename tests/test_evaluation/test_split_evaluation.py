from __future__ import annotations

import pytest

from mteb import MTEB
from tests.test_benchmark.mock_models import (
    MockSentenceTransformer,
)
from tests.test_benchmark.mock_tasks import (
    MockMultilingualRetrievalTask,
    MockMultilingualSTSTask,
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
    assert results[0].scores.keys() == {"val", "test"}


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
    assert results[0].scores.keys() == {"val"}

    results2 = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    assert "MockRetrievalTask" == results2[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"test"}
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 1
    assert results2[0].scores.keys() == {"test", "val"}


def test_no_missing_splits(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 2
    assert results[0].scores.keys() == {"test", "val"}

    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits) == 0
    assert results[0].scores.keys() == {"test", "val"}


def test_all_languages_evaluated(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "all_lang_evaluated"),
        verbosity=2,
        eval_subsets=None,
    )
    assert "MockMultilingualRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 2


def test_missing_language(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_subsets=["eng"],
    )

    assert "MockMultilingualRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]
    assert results[0].scores.keys() == {"test"}
    assert results[0].languages == ["eng"]

    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]
    assert sorted(results[0].languages) == ["eng", "fra"]
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 2


def test_no_missing_languages(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "no_missing_lang_test"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 2
    assert sorted(results[0].languages) == ["eng", "fra"]

    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "no_missing_lang_test"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits) == 0
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 2
    assert sorted(results[0].languages) == ["eng", "fra"]


def test_partial_languages(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_subsets=["fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 1
    assert results[0].languages == ["fra"]

    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_subsets=["fra", "eng"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]
    assert results[0].scores.keys() == {"test"}
    assert len(results[0].scores["test"]) == 2
    assert sorted(results[0].languages) == ["eng", "fra"]


def test_multilingual_one_missing_split_no_missing_lang(
    model, multilingual_tasks, tmp_path
):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}
    assert sorted(results[0].languages) == ["eng", "fra"]
    assert results[0].scores.keys() == {"val"}
    assert len(results[0].scores["val"]) == 2

    results = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}
    assert sorted(results[0].languages) == ["eng", "fra"]
    assert results[0].scores.keys() == {"test", "val"}
    assert len(results[0].scores["test"]) == 2
    assert len(results[0].scores["val"]) == 2


def test_multilingual_one_missing_lang_in_one_split(
    model, multilingual_tasks, tmp_path
):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}
    assert sorted(results[0].languages) == ["eng", "fra"]
    assert results[0].scores.keys() == {"val"}
    assert len(results[0].scores["val"]) == 2

    results = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_subsets=["eng"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}
    assert sorted(results[0].languages) == ["eng", "fra"]
    assert results[0].scores.keys() == {"test", "val"}
    assert len(results[0].scores["test"]) == 1
    assert len(results[0].scores["val"]) == 2

    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        eval_subsets=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}
    assert sorted(results[0].languages) == ["eng", "fra"]
    # output merged result with previous results
    assert results[0].scores.keys() == {"test", "val"}
    assert len(results[0].scores["test"]) == 2


def test_all_splits_evaluated_with_overwrite(model, tasks, tmp_path):
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "all_splits_evaluated_with_overwrite"),
        verbosity=2,
    )

    assert "MockRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 1
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"val"}
    assert results[0].scores.keys() == {"val"}

    results2 = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "all_splits_evaluated_with_overwrite"),
        verbosity=2,
        overwrite_results=True,
    )
    assert "MockRetrievalTask" == results2[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert len(last_evaluated_splits["MockRetrievalTask"]) == 2
    assert set(last_evaluated_splits["MockRetrievalTask"]) == {"val", "test"}
    assert results2[0].scores.keys() == {"val", "test"}


def test_all_splits_subsets_evaluated_with_overwrite(
    model, multilingual_tasks, tmp_path
):
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=[
            "test",
        ],
        output_folder=str(tmp_path / "all_splits_subsets_evaluated_with_overwrite"),
        verbosity=2,
        eval_subsets=["fra"],
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert results[0].scores.keys() == {"test"}
    for split in ["test"]:
        assert len(results[0].scores[split]) == 1
        assert sorted(results[0].languages) == ["fra"]

    results2 = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "all_splits_subsets_evaluated_with_overwrite"),
        verbosity=2,
        eval_subsets=["fra", "eng"],
        overwrite_results=True,
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert results2[0].scores.keys() == {"test"}
    for split in ["test"]:
        assert len(results2[0].scores[split]) == 2
        assert sorted(results2[0].languages) == ["eng", "fra"]


def test_splits_evaluated_with_prefiltering():
    """Test that the evaluation only runs on the specified languages. Issue https://github.com/embeddings-benchmark/mteb/pull/1787#issuecomment-2598205049"""
    task = MockMultilingualSTSTask().filter_languages(languages=["fra"])

    evaluation = MTEB(tasks=[task])

    results = evaluation.run(MockSentenceTransformer(), overwrite_results=True)
    result_scores = results[0].scores

    assert len(result_scores) == 1
    assert "test" in result_scores
    assert len(result_scores["test"]) == 1
    assert result_scores["test"][0]["hf_subset"] == "fra"
