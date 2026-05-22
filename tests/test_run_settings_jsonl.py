from __future__ import annotations

import json

import mteb
from mteb.cache import ResultCache
from mteb.results import TaskResult


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_save_to_cache_replaces_existing_run_settings_entry(tmp_path):
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
    assert entries[0]["split"] == "test"
    assert entries[0]["subset"] == "en"
    assert entries[0]["encode_kwargs"]["batch_size"] == 32


def test_save_to_cache_serializes_non_json_serializable_encode_kwargs(tmp_path):
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
