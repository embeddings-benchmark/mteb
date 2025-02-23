from __future__ import annotations

import sys
from pathlib import Path

import pytest

import mteb
from mteb import MTEB
from mteb.abstasks import AbsTask

from .mock_tasks import MockRetrievalText


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10 or higher")
@pytest.mark.parametrize("model", ["colbert-ir/colbertv2.0"])
@pytest.mark.parametrize("task", [MockRetrievalText()])
def test_colbert_model_e2e(task: AbsTask, model: str, tmp_path: Path):
    pytest.importorskip("pylate", reason="pylate not installed")
    eval_splits = ["test"]
    model = mteb.get_model(model)
    evaluation = MTEB(tasks=[task])

    results = evaluation.run(
        model,
        eval_splits=eval_splits,
        corpus_chunk_size=500,
        output_folder=tmp_path.as_posix(),
    )
    result = results[0]

    assert result.scores["test"][0]["ndcg_at_1"] == 1.0


def test_bm25s_e2e(tmp_path: Path):
    # fails for dataset smaller then 1000
    pytest.importorskip("bm25s", reason="bm25s not installed")
    pytest.importorskip("Stemmer", reason="PyStemmer not installed")

    model = mteb.get_model("bm25s")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])
    eval_splits = ["test"]

    evaluation = MTEB(tasks=tasks)

    results = evaluation.run(
        model, eval_splits=eval_splits, output_folder=tmp_path.as_posix()
    )
    result = results[0]

    assert result.scores["test"][0]["ndcg_at_1"] == 0.42879
