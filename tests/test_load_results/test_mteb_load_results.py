import json
import os
from pathlib import Path

import pytest

import mteb
from mteb import (
    MTEB,
)
from tests.test_load_results.conftest import (
    get_all_tasks_results,
)


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


@pytest.mark.xfail
@pytest.mark.parametrize("task", MTEB().tasks_cls)
def test_load_results_main_score_in_real_results(task):
    """Test that main score is in real results scores with equal values"""
    task_files = get_all_tasks_results()
    task_name = task.metadata.name
    result_files = task_files[task_name]
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
        assert "scores" in result.keys(), result_file + " not have 'scores'"
        for subset, subset_scores in result["scores"].items():
            assert isinstance(subset_scores, list), (
                result_file + " 'scores' is not list"
            )
            for subset_score in subset_scores:
                assert (
                    task.metadata.main_score in subset_score
                ), f"{result_file} not have {task.metadata.main_score} for task {task_name}"
                assert (
                    subset_score[task.metadata.main_score] == subset_score["main_score"]
                ), result_file


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
