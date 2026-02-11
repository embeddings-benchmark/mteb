from __future__ import annotations

import pytest

import mteb


def test_bb25_matches_bm25s():
    """bb25 with prior_weight=0.0 must produce identical rankings to bm25s."""
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    task_name = "NFCorpus"
    eval_splits = ["test"]

    bm25s_model = mteb.get_model("bm25s")
    bm25s_task = mteb.get_task(task_name, eval_splits=eval_splits)
    bm25s_results = mteb.evaluate(bm25s_model, bm25s_task, cache=None)

    bb25_model = mteb.get_model("baseline/bb25")
    bb25_task = mteb.get_task(task_name, eval_splits=eval_splits)
    bb25_results = mteb.evaluate(bb25_model, bb25_task, cache=None)

    bm25s_scores = bm25s_results[0].scores["test"][0]
    bb25_scores = bb25_results[0].scores["test"][0]

    for metric in ("ndcg_at_1", "ndcg_at_10", "map_at_10"):
        assert bb25_scores[metric] == bm25s_scores[metric], (
            f"{metric}: bb25={bb25_scores[metric]} != bm25s={bm25s_scores[metric]}"
        )
