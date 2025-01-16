from __future__ import annotations

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask

from .mock_tasks import MockRetrievalTask


@pytest.mark.parametrize("model", ["colbert-ir/colbertv2.0"])
@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_colbert_model_e2e(task: AbsTask, model: str):
    pytest.importorskip("pylate", reason="pylate not installed")
    eval_splits = ["test"]
    model = mteb.get_model(model)
    evaluation = MTEB(tasks=[task])

    results = evaluation.run(
        model,
        eval_splits=eval_splits,
        corpus_chunk_size=500,
    )
    result = results[0]

    assert result.scores["test"][0]["ndcg_at_1"] == 1.0


def test_bm25s_e2e():
    # fails for dataset smaller then 1000
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    model = mteb.get_model("bm25s")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])
    eval_splits = ["test"]

    evaluation = MTEB(tasks=tasks)

    results = evaluation.run(model, eval_splits=eval_splits)
    result = results[0]

    assert result.scores["test"][0]["ndcg_at_1"] == 0.42879
