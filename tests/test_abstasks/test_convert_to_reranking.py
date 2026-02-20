from pathlib import Path

import mteb
from mteb.cache import ResultCache
from tests.mock_tasks import MockRetrievalTask


def test_two_stage_reranking(tmp_path: Path):
    de = mteb.get_model("baseline/random-encoder-baseline")
    ce = mteb.get_model("baseline/random-cross-encoder-baseline")
    task = MockRetrievalTask()
    cache = ResultCache(tmp_path)
    results_folder = tmp_path / "retrieve_results" / "de"
    de_results = mteb.evaluate(
        de,
        task,
        overwrite_strategy="always",
        prediction_folder=results_folder,
        cache=cache,
    )
    task = task.convert_to_reranking(results_folder, top_k=1)
    assert len(task.dataset["default"]["test"]["top_ranked"]["q1"]) == 1

    ce_results = mteb.evaluate(
        ce,
        task,
        overwrite_strategy="always",
        cache=cache,
    )

    assert (
        de_results[0].scores["test"][0]["ndcg_at_1"]
        == ce_results[0].scores["test"][0]["ndcg_at_1"]
    )
