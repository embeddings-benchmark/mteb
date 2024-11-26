from __future__ import annotations

import pytest
from sentence_transformers import SentenceTransformer

import mteb
from mteb import MTEB


@pytest.fixture
def model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def nfcorpus_tasks():
    return mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])


@pytest.mark.skip(reason="WIP")
def test_all_splits_evaluated(model, nfcorpus_tasks, tmp_path):
    evaluation = MTEB(tasks=nfcorpus_tasks)
    evaluation.run(
        model,
        eval_splits=["train", "test"],
        save_predictions=True,
        output_folder=str(tmp_path / "testcase1"),
        verbosity=2,
    )
    last_evaluated_splits = evaluation.get_last_evaluated_splits()
    print(last_evaluated_splits)
    assert "NFCorpus" in last_evaluated_splits
    assert set(last_evaluated_splits["NFCorpus"]) == {"train", "test"}
    assert len(last_evaluated_splits["NFCorpus"]) == 2


@pytest.mark.skip(reason="WIP")
def test_one_missing_split(model, nfcorpus_tasks, tmp_path):
    evaluation = MTEB(tasks=nfcorpus_tasks)
    evaluation.run(
        model,
        eval_splits=["train"],
        save_predictions=True,
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    # Get model and tasks again
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nfcorpus_tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

    evaluation_2 = MTEB(tasks=nfcorpus_tasks)
    evaluation_2.run(
        model,
        eval_splits=["train", "test"],
        save_predictions=True,
        output_folder=str(tmp_path / "testcase2"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation_2.get_last_evaluated_splits()

    print(last_evaluated_splits)
    assert "NFCorpus" in last_evaluated_splits
    assert set(last_evaluated_splits["NFCorpus"]) == {"test"}
    assert len(last_evaluated_splits["NFCorpus"]) == 1


@pytest.mark.skip(reason="WIP")
def test_no_missing_splits(model, nfcorpus_tasks, tmp_path):
    evaluation_1 = MTEB(tasks=nfcorpus_tasks)
    evaluation_1.run(
        model,
        eval_splits=["train", "test"],
        save_predictions=True,
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    evaluation_2 = MTEB(tasks=nfcorpus_tasks)
    evaluation_2.run(
        model,
        eval_splits=["train", "test"],
        save_predictions=True,
        output_folder=str(tmp_path / "testcase3"),
        verbosity=2,
    )

    last_evaluated_splits = evaluation_2.get_last_evaluated_splits()

    assert "NFCorpus" in last_evaluated_splits
    assert len(last_evaluated_splits["NFCorpus"]) == 0
