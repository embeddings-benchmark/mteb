from __future__ import annotations

import json
from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.cache import ResultCache
from mteb.mocks.mock_tasks.bitext_mining import (
    MockMultilingualParallelBitextMiningTask,
)
from mteb.mocks.mock_tasks.classification import MockMultilingualClassificationTask
from mteb.results import TaskResult


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_save_to_cache_replaces_existing_run_settings_entry(tmp_path: Path):
    cache = ResultCache(cache_path=tmp_path)
    task_result = TaskResult.from_task_results(
        task=mteb.get_task("STS12"),
        scores={"test": {"en": {"main_score": 0.5}}},
        evaluation_time=100,
    )

    cache.save_to_cache(
        task_result,
        "model",
        model_revision="rev1",
        encode_kwargs={"batch_size": 16},
    )
    cache.save_to_cache(
        task_result,
        "model",
        model_revision="rev1",
        encode_kwargs={"batch_size": 32},
    )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    entries = _read_jsonl(run_settings_path)

    assert len(entries) == 1
    assert entries[0]["task"] == "STS12"
    assert entries[0]["splits"] == ["test"]
    assert entries[0]["subsets"] == ["en"]
    assert entries[0]["encode_kwargs"]["batch_size"] == 32


def test_save_to_cache_combines_subsets_with_same_settings(tmp_path: Path):
    cache = ResultCache(cache_path=tmp_path)
    for subset in ["en", "de"]:
        cache.save_to_cache(
            TaskResult.from_task_results(
                task=mteb.get_task("MassiveIntentClassification"),
                scores={"test": {subset: {"main_score": 0.5}}},
                evaluation_time=100,
            ),
            "model",
            model_revision="rev1",
            encode_kwargs={"batch_size": 16},
        )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    entries = _read_jsonl(run_settings_path)

    assert len(entries) == 1
    assert entries[0]["subsets"] == ["de", "en"]

    # rerunning a single subset with other settings splits it out again
    cache.save_to_cache(
        TaskResult.from_task_results(
            task=mteb.get_task("MassiveIntentClassification"),
            scores={"test": {"en": {"main_score": 0.5}}},
            evaluation_time=100,
        ),
        "model",
        model_revision="rev1",
        encode_kwargs={"batch_size": 32},
    )
    entries = _read_jsonl(run_settings_path)
    entries = sorted(entries, key=lambda e: e["encode_kwargs"]["batch_size"])

    assert len(entries) == 2
    assert entries[0]["subsets"] == ["de"]
    assert entries[0]["encode_kwargs"]["batch_size"] == 16
    assert entries[1]["subsets"] == ["en"]
    assert entries[1]["encode_kwargs"]["batch_size"] == 32


def test_save_to_cache_combines_splits_evaluated_on_the_same_subsets(tmp_path: Path):
    cache = ResultCache(cache_path=tmp_path)
    scores = {"en": {"main_score": 0.5}, "de": {"main_score": 0.5}}
    cache.save_to_cache(
        TaskResult.from_task_results(
            task=mteb.get_task("MassiveIntentClassification"),
            scores={"test": scores, "validation": scores},
            evaluation_time=100,
        ),
        "model",
        model_revision="rev1",
        encode_kwargs={"batch_size": 16},
    )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    entries = _read_jsonl(run_settings_path)

    assert len(entries) == 1
    assert entries[0]["splits"] == ["test", "validation"]
    assert entries[0]["subsets"] == ["de", "en"]

    # a subset evaluated on a single split only breaks the shared row apart
    cache.save_to_cache(
        TaskResult.from_task_results(
            task=mteb.get_task("MassiveIntentClassification"),
            scores={"test": {"fr": {"main_score": 0.5}}},
            evaluation_time=100,
        ),
        "model",
        model_revision="rev1",
        encode_kwargs={"batch_size": 16},
    )
    entries = sorted(_read_jsonl(run_settings_path), key=lambda e: e["splits"])

    assert len(entries) == 2
    assert entries[0]["splits"] == ["test"]
    assert entries[0]["subsets"] == ["de", "en", "fr"]
    assert entries[1]["splits"] == ["validation"]
    assert entries[1]["subsets"] == ["de", "en"]


def test_save_to_cache_serializes_non_json_serializable_encode_kwargs(tmp_path: Path):
    cache = ResultCache(cache_path=tmp_path)
    task_result = TaskResult.from_task_results(
        task=mteb.get_task("STS13"),
        scores={"test": {"default": {"main_score": 0.5}}},
        evaluation_time=100,
    )

    cache.save_to_cache(
        task_result,
        "model",
        model_revision="rev1",
        encode_kwargs={"custom": object()},
    )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    entries = _read_jsonl(run_settings_path)

    assert len(entries) == 1
    assert isinstance(entries[0]["encode_kwargs"]["custom"], str)


@pytest.mark.parametrize(
    ("task_cls", "expected_batch_sizes"),
    [
        (MockMultilingualClassificationTask, (8, 16)),
        (MockMultilingualParallelBitextMiningTask, (8, 16)),
    ],
    ids=["classification", "parallel-bitext"],
)
def test_resume_records_settings_for_evaluated_subsets(
    tmp_path: Path, task_cls: type[AbsTask], expected_batch_sizes: tuple[int, int]
):
    model = mteb.get_model("mteb/baseline-random-encoder")
    cache = ResultCache(cache_path=tmp_path)
    task = task_cls()
    first_run_task = task_cls().filter_languages(
        languages=None, hf_subsets=task.hf_subsets[:1]
    )
    mteb.evaluate(model, first_run_task, cache=cache, encode_kwargs={"batch_size": 8})
    mteb.evaluate(model, task, cache=cache, encode_kwargs={"batch_size": 16})

    (run_settings_path,) = tmp_path.rglob("run_settings.jsonl")
    batch_sizes = {
        subset: entry["encode_kwargs"]["batch_size"]
        for entry in _read_jsonl(run_settings_path)
        for subset in entry["subsets"]
    }
    assert batch_sizes == dict(zip(task.hf_subsets, expected_batch_sizes, strict=True))


def test_save_to_cache_records_only_evaluated_splits(tmp_path: Path):
    cache = ResultCache(cache_path=tmp_path)
    score = {"main_score": 0.5}
    cache.save_to_cache(
        TaskResult.from_task_results(
            task=MockMultilingualClassificationTask(),
            scores={"test": {"eng": score, "fra": score}, "validation": {"eng": score}},
            evaluation_time=0,
        ),
        "model",
        model_revision="rev1",
        evaluated_splits={"test": ["fra"]},
    )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    (entry,) = _read_jsonl(run_settings_path)
    assert entry["splits"] == ["test"]
    assert entry["subsets"] == ["fra"]
