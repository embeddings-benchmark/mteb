from __future__ import annotations

import pytest

import mteb


@pytest.mark.parametrize("model_name", ["bm25s", "baseline/bb25"])
def test_bm25_e2e(model_name):
    # fails for dataset smaller than 1000
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    model = mteb.get_model(model_name)
    task = mteb.get_task("NFCorpus", eval_splits=["test"])

    results = mteb.evaluate(model, task, cache=None)
    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.42879
