import json
from pathlib import Path

import pytest

import mteb
from mteb.mocks.mock_tasks import MockRerankingTask


@pytest.mark.parametrize("candidates", ["missing", "empty"])
def test_random_colbert_reranking_query_without_candidates(
    tmp_path: Path, candidates: str
) -> None:
    """A query without reranking candidates gets empty results instead of an error."""
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

    assert set(predictions) == {"q1", "q2"}
    assert set(predictions["q1"]) == {"d1", "d2"}
    assert predictions["q2"] == {}
