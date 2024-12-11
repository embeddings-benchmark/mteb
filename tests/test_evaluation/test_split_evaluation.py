from __future__ import annotations

import pytest

from mteb import MTEB
from tests.test_benchmark.mock_models import (
    MockSentenceTransformer,
)
from tests.test_benchmark.mock_tasks import (
    MockMultilingualRetrievalTask, MockRetrievalTask,
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
        eval_langs=None,  # means run all languages
    )
    assert "MockMultilingualRetrievalTask" == results[0].task_name
    # Since we ran all subsets, last_evaluated_splits should contain "test"
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    # Only one split 'test' but multiple languages were evaluated
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_missing_language(model, multilingual_tasks, tmp_path):
    # First, run only English
    evaluation = MTEB(tasks=multilingual_tasks)
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_langs=["eng"],  # run only English language subsets
    )

    assert "MockMultilingualRetrievalTask" == results[0].task_name
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # We evaluated 'test' split for 'eng'
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]

    results2 = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "missing_lang_test"),
        verbosity=2,
        eval_langs=["eng", "fra"],  # now we include German
        overwrite_results=True,
    )

    # Check that only missing subset (German) was run this time
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # We should see that only the missing language (fra) was evaluated now
    # Since we ran with overwrite_results and included eng (already done) + fra (new),
    # only the new language subsets are run. This means "test" split again, but now
    # effectively for the missing subset.
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_no_missing_languages(model, multilingual_tasks, tmp_path):
    # Run on all languages once
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
    # One split fully evaluated
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1

    # Run again with the same languages and overwrite_results=True
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
    # Since nothing is missing and we overwrite, we re-run everything,
    # but since we already have the data and re-ran, no missing subsets after the run
    # means last_evaluated_splits is empty this time.
    assert "MockMultilingualRetrievalTask" in last_evaluated_splits
    # If no subsets were missing, it means we had to run again due to overwrite,
    # after running again, no "new" subsets were missing, so no "new" splits get recorded.
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 0


def test_partial_languages(model, multilingual_tasks, tmp_path):
    # Run first only French
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_langs=["fra"],  # run only French first
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # Only French run now, so "test" is recorded
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]

    # Now run French and English
    # French is done, so only English should run now
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "partial_lang_test"),
        verbosity=2,
        eval_langs=["fra", "eng"],
        overwrite_results=True,
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # After this run, English was missing previously (since we only did French),
    # Now we evaluate English. "test" again is recorded.
    assert len(last_evaluated_splits["MockMultilingualRetrievalTask"]) == 1
    assert last_evaluated_splits["MockMultilingualRetrievalTask"] == ["test"]


def test_multilingual_multiple_splits_partial_langs_partial_splits(model, multilingual_tasks, tmp_path):
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        langs_to_run=["eng", "fra"],  # run only English and French
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # Only val split was run, with eng and fra
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}

    # Now run 'val' and 'test' splits on all languages eng, deu, fra
    # English and French on val are done, but we haven't done German on val,
    # and none of the languages done on 'test' yet.
    # So this should run German on 'val', and eng, deu, fra on 'test'
    # because test is completely missing.
    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "partial_langs_partial_splits"),
        verbosity=2,
        langs_to_run=["eng", "fra"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # This time:
    # - For 'val' split: only German was missing. So 'val' will appear in last_evaluated_splits again.
    # - For 'test' split: all languages (eng, deu, fra) were missing, so 'test' will also appear again.
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val", "test"}


def test_multilingual_multiple_splits_missing_only_one_language_in_one_split(model, multilingual_tasks, tmp_path):
    # Run all languages on 'val' only
    evaluation = MTEB(tasks=multilingual_tasks)
    _ = evaluation.run(
        model,
        eval_splits=["val"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        langs_to_run=["eng", "fra"],
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # val split fully done for all languages
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"val"}

    # Now run 'val' and 'test' but only English and German (no French)
    # val is fully done for eng, deu, fra. So no missing subsets for val.
    # test not run yet, so test for eng, deu will run now.
    _ = evaluation.run(
        model,
        eval_splits=["val", "test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        langs_to_run=["eng"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # Since val was already fully done for all languages, no new evaluation for val
    # test was missing completely, but we only requested eng and deu now, so we run them
    # last_evaluated_splits should show that test was run this time
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}

    # Now run again 'test' for all languages, including French
    # eng, deu done on test, fra missing on test
    _ = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=str(tmp_path / "one_lang_one_split"),
        verbosity=2,
        langs_to_run=["eng", "fra"],
        overwrite_results=True,
    )

    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    # Now only French on test was missing previously, so we run it now.
    assert set(last_evaluated_splits["MockMultilingualRetrievalTask"]) == {"test"}