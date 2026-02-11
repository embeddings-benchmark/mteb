from __future__ import annotations

import pytest

import mteb


def test_bb25_e2e():
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    model = mteb.get_model("baseline/bb25")
    task = mteb.get_task("NFCorpus", eval_splits=["test"])

    results = mteb.evaluate(model, task, cache=None)
    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.42879
