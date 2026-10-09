"""Prepared candidates use the existing model experiment storage and exports."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import mteb
from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
from mteb.cache import ResultCache
from mteb.mocks.mock_tasks import MockAggregatedTask, MockRetrievalTask


def predictions(path: Path, *, reverse: bool = False):
    good, bad = ({"d1": 2.0, "d2": 1.0}, {"d2": 2.0, "d1": 1.0})
    values = {"q1": bad if reverse else good, "q2": good if reverse else bad}
    data = {
        "mteb_model_meta": {"model_name": "test/retriever", "revision": "abc"},
        "default": dict.fromkeys(MockRetrievalTask().eval_splits, values),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def task_with_sources(tmp_path: Path):
    task = MockRetrievalTask()
    first, second = tmp_path / "bm25.json", tmp_path / "qwen.json"
    predictions(first)
    predictions(second, reverse=True)
    task.first_stage_predictions = {"bm25": first, "qwen": second}
    return task


def evaluate(model, task, cache, **kwargs: Any):
    return mteb.evaluate(model, task, cache=cache, co2_tracker=False, **kwargs)


def test_named_local_and_directory_sources_preserve_candidates_and_qrels(tmp_path):
    task = task_with_sources(tmp_path)
    task.load_data()
    data = task.dataset["default"]["test"]
    corpus, queries, qrels = (
        data["corpus"],
        data["queries"],
        deepcopy(data["relevant_docs"]),
    )
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    assert data["top_ranked"] == {"q1": ["d1"], "q2": ["d2"]}
    assert data["corpus"] is corpus and data["queries"] is queries
    assert data["relevant_docs"] == qrels
    named = task._reranking_experiment
    task.convert_to_reranking(tmp_path / "bm25.json", top_k=1)
    assert task._reranking_experiment["predictions"] == named["predictions"]
    assert task._reranking_experiment["name"] == "local"
    (tmp_path / task.prediction_file_name).write_bytes(
        (tmp_path / "bm25.json").read_bytes()
    )
    task.convert_to_reranking(tmp_path, top_k=1)
    assert task._reranking_experiment["predictions"] == named["predictions"]


def test_pinned_hub_source_and_tied_scores(tmp_path, monkeypatch):
    path = tmp_path / "predictions.json"
    data = predictions(path)
    for split in data["default"].values():
        split["q1"] = {"d2": 1.0, "d1": 1.0}
    path.write_text(json.dumps(data))
    calls = []

    def download(**kwargs: Any):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download", download
    )
    task = MockRetrievalTask()
    task.first_stage_predictions = {
        "frozen": FirstStagePredictionSource("org/predictions", "bm25.json", "a" * 40)
    }
    task.convert_to_reranking(first_stage="frozen", top_k=2)
    assert task.dataset["default"]["test"]["top_ranked"]["q1"] == ["d2", "d1"]
    assert calls == [
        {
            "repo_id": "org/predictions",
            "filename": "bm25.json",
            "revision": "a" * 40,
            "repo_type": "dataset",
        }
    ]
    with pytest.raises(ValueError, match="commit SHA"):
        FirstStagePredictionSource("org/predictions", "bm25.json", "main")


def test_experiments_isolate_sources_depth_and_content_without_mutating_model(
    tmp_path, monkeypatch
):
    task = task_with_sources(tmp_path)
    model = mteb.get_model("mteb/baseline-random-encoder", test_param="retained")
    original_meta = model.mteb_model_meta.model_dump()
    cache = ResultCache(tmp_path / "cache")
    results = [evaluate(model, task, cache)]
    for name in ("bm25", "qwen"):
        for depth in (1, 2):
            task.convert_to_reranking(first_stage=name, top_k=depth)
            result = evaluate(
                model, task, cache, prediction_folder=tmp_path / "predictions"
            )
            assert result.model_meta.experiment_kwargs["test_param"] == "retained"
            assert (
                result.model_meta.experiment_kwargs["first_stage"]
                == task._reranking_experiment
            )
            assert (
                tmp_path
                / "predictions"
                / "experiments"
                / result.experiment_name
                / task.prediction_file_name
            ).exists()
            assert result[0].scores["test"][0]["previous_results_model_meta"] == {
                "model_name": "test/retriever",
                "revision": "abc",
            }
            results.append(result)
    predictions(tmp_path / "bm25.json", reverse=True)
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    results.append(evaluate(model, task, cache))
    assert len({r.experiment_name for r in results}) == 6
    assert model.mteb_model_meta.model_dump() == original_meta
    loaded = cache.load_results(
        models=[r.model_meta for r in results], include_remote=False
    )
    assert len(loaded.model_results) == 6
    assert len(loaded.join_revisions().model_results) == 6
    rows = loaded._to_dataset()
    assert len(rows) == 6 * len(task.eval_splits)
    assert all(row["experiments"]["test_param"] == "retained" for row in rows)
    for result in results:
        assert (
            cache.load_task_result(task.metadata.name, result.model_meta).get_score()
            == result[0].get_score()
        )

    def no_inference(*args: Any, **kwargs: Any):
        raise AssertionError("Expected a cache hit")

    monkeypatch.setattr(model, "encode", no_inference)
    assert (
        evaluate(model, task, cache, overwrite_strategy="only-cache")[0].get_score()
        == results[-1][0].get_score()
    )
    assert (
        evaluate(model, MockRetrievalTask(), cache, overwrite_strategy="only-cache")[
            0
        ].get_score()
        == results[0][0].get_score()
    )


def test_model_meta_loader_does_not_receive_first_stage_context(tmp_path):
    task = task_with_sources(tmp_path).convert_to_reranking(first_stage="bm25", top_k=1)
    model = mteb.get_model("mteb/baseline-random-encoder")
    meta = model.mteb_model_meta
    calls = []

    def loader(name, **kwargs: Any):
        calls.append(kwargs)
        assert "first_stage" not in kwargs
        return model

    meta = meta.model_copy(update={"loader": loader})
    result = evaluate(meta, task, ResultCache(tmp_path / "cache"))
    assert len(calls) == 1
    assert meta.experiment_kwargs is None
    assert model.mteb_model_meta.experiment_kwargs is None
    assert result.model_meta.experiment_kwargs["first_stage"]["name"] == "bm25"
    rows = mteb.BenchmarkResults(model_results=[result])._to_dataset()
    assert rows[0]["experiments"]["first_stage"]["top_k"] == 1
    saved_meta = result.model_meta
    reloaded_model = saved_meta.load_model()
    assert reloaded_model.mteb_model_meta.experiment_kwargs is None
    ordinary = evaluate(
        saved_meta, MockRetrievalTask(), ResultCache(tmp_path / "cache")
    )
    assert ordinary.experiment_name is None
    assert ordinary[0].scores["test"][0].get("previous_results_model_meta") is None
    assert saved_meta.experiment_kwargs["first_stage"]["name"] == "bm25"


def test_domains_share_a_pinned_experiment_and_pin_or_file_changes_isolate_it(
    tmp_path, monkeypatch
):
    local = tmp_path / "predictions.json"
    predictions(local)
    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download",
        lambda **kwargs: str(local),
    )
    tasks = []
    for domain in ("One", "Two"):
        task = MockRetrievalTask()
        task.metadata = task.metadata.model_copy(update={"name": f"MockDomain{domain}"})
        task.first_stage_predictions = {
            "bm25": FirstStagePredictionSource(
                "org/preds", f"bm25/text/{task.prediction_file_name}", "a" * 40
            )
        }
        tasks.append(task.convert_to_reranking(first_stage="bm25", top_k=1))
    model = mteb.get_model("mteb/baseline-random-encoder")
    cache = ResultCache(tmp_path / "cache")
    result = evaluate(model, iter(tasks), cache)
    assert len(result) == 2
    assert (
        result.model_meta.experiment_kwargs["first_stage"]["predictions"]["filename"]
        == "bm25/text/{task}_predictions.json"
    )
    assert (
        len(
            cache.load_results(
                models=[result.model_meta],
                include_remote=False,
                validate_and_filter=False,
            )[0]
        )
        == 2
    )
    for filename, revision in [
        (f"bm25/text/{tasks[0].prediction_file_name}", "b" * 40),
        (f"bm25_text/{tasks[0].prediction_file_name}", "a" * 40),
        ("bm25/alternative.json", "a" * 40),
    ]:
        tasks[0].first_stage_predictions = {
            "bm25": FirstStagePredictionSource("org/preds", filename, revision)
        }
        tasks[0].convert_to_reranking(first_stage="bm25", top_k=1)
        assert (
            evaluate(model, tasks[0], cache).experiment_name != result.experiment_name
        )


def test_mixed_experiments_are_rejected_before_evaluation(tmp_path):
    task = task_with_sources(tmp_path).convert_to_reranking(first_stage="bm25", top_k=1)
    tasks = [task, MockRetrievalTask()]
    aggregate = MockAggregatedTask()
    aggregate.metadata = aggregate.metadata.model_copy(update={"tasks": tasks})
    for input_tasks in (tasks, aggregate):
        with pytest.raises(ValueError, match="different first-stage"):
            evaluate(mteb.get_model("mteb/baseline-random-encoder"), input_tasks, None)


def test_failed_conversion_preserves_candidates_and_context(tmp_path):
    task = task_with_sources(tmp_path).convert_to_reranking(first_stage="bm25", top_k=1)
    before = deepcopy(task.dataset["default"]["test"]["top_ranked"])
    context = task._reranking_experiment
    data = json.loads((tmp_path / "qwen.json").read_text())
    data["default"][list(task.dataset["default"])[-1]] = []
    (tmp_path / "qwen.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dictionary"):
        task.convert_to_reranking(first_stage="qwen", top_k=2)
    assert task.dataset["default"]["test"]["top_ranked"] == before
    assert task._reranking_experiment == context
    assert task._top_k == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Supply top_ranked_path or first_stage"),
        ({"top_ranked_path": "bm25.json", "first_stage": "bm25"}, "not both"),
        ({"first_stage": "missing"}, "Available sources:.*bm25.*qwen"),
        ({"first_stage": "bm25", "top_k": 0}, "positive"),
    ],
)
def test_source_selection_errors(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        task_with_sources(tmp_path).convert_to_reranking(**kwargs)


def test_positional_string_remains_a_path(tmp_path, monkeypatch):
    task = task_with_sources(tmp_path)
    predictions(tmp_path / "bm25", reverse=True)
    monkeypatch.chdir(tmp_path)
    task.convert_to_reranking("bm25", 1)
    assert task.dataset["default"]["test"]["top_ranked"]["q1"] == ["d2"]
    assert task._reranking_experiment["name"] == "local"
