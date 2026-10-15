import json
from pathlib import Path

import mteb
from mteb.cache import ResultCache
from mteb.mocks.mock_tasks import MockRetrievalTask


def _ranking(prediction_folder: Path, task) -> dict[str, list[str]]:
    with task._predictions_path(prediction_folder).open() as f:
        predictions = json.load(f)["default"]["test"]
    return {
        query_id: sorted(scores, key=scores.get, reverse=True)
        for query_id, scores in predictions.items()
    }


def test_no_reranking_keeps_first_stage_ranking(tmp_path: Path):
    first_stage = mteb.get_model("mteb/baseline-random-encoder")
    reranker = mteb.get_model("mteb/baseline-no-reranking")
    task = MockRetrievalTask()
    cache = ResultCache(tmp_path)

    first_stage_folder = tmp_path / "first_stage"
    mteb.evaluate(
        first_stage,
        task,
        overwrite_strategy="always",
        prediction_folder=first_stage_folder,
        cache=cache,
    )
    task = task.convert_to_reranking(first_stage_folder)

    reranked_folder = tmp_path / "reranked"
    mteb.evaluate(
        reranker,
        task,
        overwrite_strategy="always",
        prediction_folder=reranked_folder,
        cache=cache,
    )

    assert _ranking(reranked_folder, task) == _ranking(first_stage_folder, task)
