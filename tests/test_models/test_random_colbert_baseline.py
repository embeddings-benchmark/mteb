import json
from pathlib import Path

import pytest

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask


@pytest.mark.parametrize("candidates", ["missing", "empty"])
def test_random_colbert_reranking_query_without_candidates(
    tmp_path: Path, candidates: str
) -> None:
    """Queries without reranking candidates are excluded before evaluation."""
    task = MockRerankingTask()
    task.load_data()
    top_ranked = task.dataset["default"]["test"]["top_ranked"]
    if candidates == "missing":
        del top_ranked["q2"]
    else:
        top_ranked["q2"] = []
    model = mteb.get_model("mteb/baseline-random-colbert")

    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open(encoding="utf-8") as f:
        predictions = json.load(f)["default"]["test"]

    assert set(predictions) == {"q1"}
    assert set(predictions["q1"]) == {"d1", "d2"}
    assert set(task.dataset["default"]["test"]["queries"]["id"]) == {"q1"}
    assert set(task.dataset["default"]["test"]["relevant_docs"]) == {"q1"}
