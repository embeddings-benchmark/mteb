from __future__ import annotations

import json

from mteb.cache import ResultCache


class DummyTaskResult:
    def __init__(self, task_name: str, scores: dict[str, list[dict[str, str]]]) -> None:
        self.task_name = task_name
        self.scores = scores

    @staticmethod
    def to_disk(path) -> None:
        path.write_text("{}", encoding="utf-8")


class UnserializableValue:
    def __str__(self) -> str:
        return "unserializable-value"


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_save_to_cache_replaces_existing_run_settings_entry(tmp_path, monkeypatch):
    cache = ResultCache(cache_path=tmp_path)
    task_result = DummyTaskResult(
        task_name="STS12",
        scores={"test": [{"hf_subset": "en"}]},
    )

    monkeypatch.setattr(
        "mteb.cache.result_cache.get_package_versions",
        lambda: {"mteb": "1.0.0", "torch": None},
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


def test_save_to_cache_serializes_non_json_serializable_encode_kwargs(
    tmp_path, monkeypatch
):
    cache = ResultCache(cache_path=tmp_path)
    task_result = DummyTaskResult(
        task_name="STS13",
        scores={"test": [{"hf_subset": "default"}]},
    )

    monkeypatch.setattr(
        "mteb.cache.result_cache.get_package_versions",
        lambda: {"mteb": "1.0.0"},
    )

    cache.save_to_cache(
        task_result,
        "model",
        model_revision="rev1",
        encode_kwargs={"custom": UnserializableValue()},
    )

    run_settings_path = tmp_path / "results" / "model" / "rev1" / "run_settings.jsonl"
    entries = _read_jsonl(run_settings_path)

    assert len(entries) == 1
    assert entries[0]["encode_kwargs"]["custom"] == "unserializable-value"
