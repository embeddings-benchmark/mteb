import os
from pathlib import Path

import pytest

import mteb


def test_mteb_load_results():
    tests_path = Path(__file__).parent.parent

    os.environ["MTEB_CACHE"] = str(tests_path)

    results = mteb.load_results(download_latest=False)

    assert isinstance(results, dict)
    for model in results:
        assert isinstance(results[model], dict)
        for revision in results[model]:
            assert isinstance(results[model][revision], list)
            for result in results[model][revision]:
                assert isinstance(result, mteb.MTEBResults)

    known_model = "sentence-transformers/average_word_embeddings_levy_dependency"
    known_revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    assert known_model in results
    assert known_revision in results[known_model]


@pytest.mark.xfail(reason="Some models results have wrong main score")
def test_load_results_scores():
    """Test that all keys from actual task results presented in real task result"""
    results = mteb.load_results()

    for model, revscores in results.items():
        for rev, mtebscore in revscores.items():
            for mteb_result in mtebscore:
                task = mteb.get_task(mteb_result.task_name)
                score = mteb_result.get_score(
                    getter=lambda scores: scores[task.metadata.main_score]
                )
                main_score = mteb_result.get_score(
                    getter=lambda scores: scores["main_score"]
                )
                assert score == main_score, f"{model} {mteb_result.task_name}"
