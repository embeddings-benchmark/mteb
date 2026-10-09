"""Named candidates keep their identity through evaluation, storage and export."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import mteb
from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
from mteb.cache import ResultCache
from mteb.cache.result_cache import CopyResultsAction
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
    task.load_data()
    task.dataset["default"] = {
        split: dict(data) for split, data in task.dataset["default"].items()
    }
    first = tmp_path / "bm25.json"
    second = tmp_path / "qwen.json"
    predictions(first)
    predictions(second, reverse=True)
    task.first_stage_predictions = {"bm25": first, "qwen": second}
    return task


def evaluate(model, task, cache, **kwargs: Any):
    return mteb.evaluate(model, task, cache=cache, co2_tracker=False, **kwargs)


def test_named_and_local_sources_preserve_order_and_qrels(tmp_path):
    task = task_with_sources(tmp_path)
    shared = task.dataset["default"]["test"]
    corpus, queries, qrels = (
        shared["corpus"],
        shared["queries"],
        deepcopy(shared["relevant_docs"]),
    )
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    assert shared["top_ranked"] == {"q1": ["d1"], "q2": ["d2"]}
    assert shared["corpus"] is corpus and shared["queries"] is queries
    assert shared["relevant_docs"] == qrels
    assert task.hf_subsets == ["default"]
    named = task.reranking_configuration
    task.convert_to_reranking(tmp_path / "bm25.json", top_k=1)
    assert task.reranking_configuration.predictions == named.predictions
    assert task.reranking_configuration.first_stage == "local"
    (tmp_path / task.prediction_file_name).write_bytes(
        (tmp_path / "bm25.json").read_bytes()
    )
    task.convert_to_reranking(tmp_path, top_k=1)
    assert task.reranking_configuration.predictions == named.predictions


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
        "frozen": FirstStagePredictionSource(
            "org/predictions", "bm25.json", "a" * 40, "text"
        )
    }
    task.convert_to_reranking(first_stage="frozen", top_k=2)
    assert task.dataset["default"]["test"]["top_ranked"]["q1"] == ["d2", "d1"]
    assert task.reranking_configuration.predictions.revision == "a" * 40
    assert task.reranking_configuration.document_representation == "text"
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


def test_cache_isolates_candidates_depth_content_and_model_experiments(
    tmp_path, monkeypatch
):
    task = task_with_sources(tmp_path)
    model = mteb.get_model("mteb/baseline-random-encoder", test_param="retained")
    original_meta = model.mteb_model_meta.model_dump()
    cache = ResultCache(tmp_path / "cache")
    output = tmp_path / "predictions"
    results = [evaluate(model, task, cache)]
    for name, depth in [("bm25", 1), ("qwen", 1), ("bm25", 2)]:
        task.convert_to_reranking(first_stage=name, top_k=depth)
        result = evaluate(model, task, cache, prediction_folder=output)
        assert result.model_meta.experiment_kwargs == {"test_param": "retained"}
        assert result[0].reranking == task.reranking_configuration
        assert "reranking" not in result[0].scores["test"][0]
        assert (
            output
            / "experiments"
            / result.experiment_name
            / "reranking"
            / result[0].reranking.configuration_id
            / task.prediction_file_name
        ).exists()
        results.append(result)
    predictions(tmp_path / "bm25.json", reverse=True)
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    results.append(evaluate(model, task, cache))
    assert (
        len(
            {
                r[0].reranking.configuration_id if r[0].reranking else None
                for r in results
            }
        )
        == 5
    )
    assert len({r.experiment_name for r in results}) == 1
    assert model.mteb_model_meta.model_dump() == original_meta
    for result in results:
        restored = cache.load_task_result(
            task.metadata.name, result.model_meta, reranking=result[0].reranking
        )
        assert restored.reranking == result[0].reranking
        assert restored.get_score() == result[0].get_score()
    loaded = cache.load_results(
        models=[result.model_meta], include_remote=False, validate_and_filter=False
    )
    assert len(loaded[0].task_results) == 5

    def no_inference(*args: Any, **kwargs: Any):
        raise AssertionError("Expected a cache hit")

    monkeypatch.setattr(model, "encode", no_inference)
    assert (
        evaluate(model, task, cache, overwrite_strategy="only-cache")[0].reranking
        == results[-1][0].reranking
    )
    assert (
        evaluate(model, MockRetrievalTask(), cache, overwrite_strategy="only-cache")[
            0
        ].reranking
        is None
    )


def test_two_domains_three_sources_survive_submission_and_export(tmp_path, monkeypatch):
    local = tmp_path / "predictions.json"
    predictions(local)
    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download",
        lambda **kwargs: str(local),
    )
    model = mteb.get_model("mteb/baseline-random-encoder")
    cache = ResultCache(tmp_path / "cache")
    tasks = []
    for domain in ("One", "Two"):
        for name, modality in [
            ("bm25", "text"),
            ("bge", "text"),
            ("qwen", "text-image"),
        ]:
            task = MockRetrievalTask()
            task.metadata = task.metadata.model_copy(
                update={"name": f"MockDomain{domain}"}
            )
            task.first_stage_predictions = {
                name: FirstStagePredictionSource(
                    "org/preds", f"{name}/{modality}/{domain}.json", "a" * 40, modality
                )
            }
            task.convert_to_reranking(first_stage=name, top_k=1)
            tasks.append(task)
    result = evaluate(model, tasks, cache)
    assert result.model_meta == model.mteb_model_meta
    loaded = cache.load_results(
        include_remote=False, validate_and_filter=False, only_main_score=True
    )
    assert len(loaded[0].task_results) == 6
    assert len(loaded.join_revisions()[0].task_results) == 6
    exported = loaded._to_dataset()
    assert len(set(exported["reranking_id"])) == 3
    assert len({(row["task_name"], row["reranking_id"]) for row in exported}) == 6
    assert all(
        row["reranking"]["predictions"]["revision"] == "a" * 40 for row in exported
    )
    assert all(
        row["previous_results_model_meta"]
        == {"model_name": "test/retriever", "revision": "abc"}
        for row in exported
    )
    for cfg in set(exported["reranking_id"]):
        selected = loaded.filter_reranking(cfg)
        assert len(selected[0].task_results) == 2
        assert len(selected.to_dataframe()) == 2
    for summarise in (
        loaded.to_dataframe,
        loaded.get_aggregated_scores,
        loaded[0]._get_scores,
    ):
        with pytest.raises(ValueError, match="Select one reranking configuration"):
            summarise()
    with pytest.raises(ValueError, match="different first-stage"):
        result[0].merge(result[1])
    assert result[0].merge(result[0]).reranking == result[0].reranking

    remote = ResultCache(tmp_path / "submitted")
    action = CopyResultsAction(
        cache._get_unsubmitted_results([model.mteb_model_meta]),
        remote.cache_path / "results",
    )
    action.do()
    restored = remote.load_results(
        models=[model.mteb_model_meta.name],
        include_remote=False,
        validate_and_filter=False,
    )
    assert len(restored[0].task_results) == 6
    assert set(restored._to_dataset()["reranking_id"]) == set(exported["reranking_id"])
    action.undo()
    assert not remote.get_cache_paths(include_remote=False)


def test_pinned_artifact_changes_cannot_reuse_cached_metrics(tmp_path, monkeypatch):
    local = tmp_path / "predictions.json"
    predictions(local)
    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download",
        lambda **kwargs: str(local),
    )
    task = MockRetrievalTask()
    task.first_stage_predictions = {
        "bm25": FirstStagePredictionSource("org/preds", "bm25/task.json", "a" * 40)
    }
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    cache = ResultCache(tmp_path / "cache")
    model = mteb.get_model("mteb/baseline-random-encoder")
    first = evaluate(model, task, cache)[0]
    predictions(local, reverse=True)
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    assert (
        first.reranking.configuration_id
        == task.reranking_configuration.configuration_id
    )
    with pytest.raises(ValueError, match="provenance differs"):
        evaluate(model, task, cache, overwrite_strategy="only-cache")
    task.first_stage_predictions["bm25"] = FirstStagePredictionSource(
        "org/preds", "bm25/task.json", "b" * 40
    )
    task.convert_to_reranking(first_stage="bm25", top_k=1)
    assert (
        first.reranking.configuration_id
        != task.reranking_configuration.configuration_id
    )


def test_aggregate_requires_explicit_candidate_policy(tmp_path):
    task = task_with_sources(tmp_path).convert_to_reranking(first_stage="bm25", top_k=1)
    aggregate = MockAggregatedTask()
    aggregate.metadata = aggregate.metadata.model_copy(update={"tasks": [task]})
    with pytest.raises(ValueError, match="Evaluate converted child tasks"):
        evaluate(mteb.get_model("mteb/baseline-random-encoder"), aggregate, None)


def test_failed_conversion_preserves_candidates_and_context(tmp_path):
    task = task_with_sources(tmp_path).convert_to_reranking(first_stage="bm25", top_k=1)
    before = deepcopy(task.dataset["default"]["test"]["top_ranked"])
    provenance = task.reranking_configuration
    path = tmp_path / "qwen.json"
    data = json.loads(path.read_text())
    data["default"][list(task.dataset["default"])[-1]] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dictionary"):
        task.convert_to_reranking(first_stage="qwen", top_k=2)
    assert task.dataset["default"]["test"]["top_ranked"] == before
    assert task.reranking_configuration == provenance
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
    assert task.reranking_configuration.first_stage == "local"
