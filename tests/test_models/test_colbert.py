from __future__ import annotations

from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.mock_tasks import MockRetrievalTask


@pytest.mark.parametrize("model_name", ["colbert-ir/colbertv2.0"])
@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_colbert_model_e2e(task: AbsTask, model_name: str, tmp_path: Path):
    pytest.importorskip("pylate", reason="pylate not installed")
    task._eval_splits = ["test"]

    model = mteb.get_model(model_name)
    results = mteb.evaluate(model, task, cache=None)

    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.0
