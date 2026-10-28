"""Disk-cache readers see one complete snapshot under I/O failures and overlap."""

from __future__ import annotations

import io
import json
import struct
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mteb.api import frames
from mteb.api.settings import Settings

REPO = "test/results"
SNAPSHOT_FILE = "api-cache-v4.zip"


def _frame(score: float | None) -> pl.DataFrame:
    schema = {
        "model_name": pl.String,
        "task_name": pl.String,
        "split": pl.String,
        "subset": pl.String,
        "score": pl.Float64,
    }
    if score is None:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(
        {
            "model_name": ["test/model"],
            "task_name": ["test-task"],
            "split": ["test"],
            "subset": ["default"],
            "score": [score],
        },
        schema=schema,
    )


def _write_generation(value: float, *, second_empty: bool = False) -> None:
    frames._write_disk_cache(
        REPO,
        f"sha-{value:g}",
        {
            "first": _frame(value),
            "second": _frame(None if second_empty else value + 0.25),
        },
        _frame(value + 0.5),
    )


def _scores(snapshot) -> tuple[float, float | None, float]:
    assert snapshot is not None
    loaded, unified = snapshot
    second = loaded.get("second")
    return (
        loaded["first"]["score"].item(),
        None if second is None or second.is_empty() else second["score"].item(),
        unified["score"].item(),
    )


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "leaderboard"
    directory.mkdir()
    monkeypatch.setattr(frames, "_disk_cache_dir", lambda: directory)
    return directory


def test_disk_snapshot_round_trip_with_empty_owner(cache_dir: Path):
    loaded = {"owner / %中文": _frame(0.3), "empty": _frame(None)}
    unified = _frame(0.8)  # The views can legitimately have different scores.
    frames._write_disk_cache(REPO, "old-sha", loaded, unified)

    result = frames._read_disk_cache(REPO, "old-sha")

    assert result is not None
    restored, restored_unified = result
    assert_frame_equal(restored["owner / %中文"], loaded["owner / %中文"])
    assert "empty" not in restored or restored["empty"].is_empty()
    assert_frame_equal(restored_unified, unified)
    assert frames._read_disk_cache(REPO, None) is not None
    assert frames._read_disk_cache(REPO, "new-sha") is None
    assert frames._read_disk_cache("another/repo", None) is None


def test_failed_second_frame_keeps_previous_complete_snapshot(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    original = pl.DataFrame.write_parquet
    writes = 0

    def fail_second_frame(self, *args: Any, **kwargs: Any):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected second-frame write failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_second_frame)
    _write_generation(9.0)

    assert _scores(frames._read_disk_cache(REPO, None)) == (1.0, 1.25, 1.5)
    assert list(cache_dir.glob("*.tmp")) == []


def test_failed_publication_keeps_previous_complete_snapshot(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)

    def cannot_replace(self, target):
        raise PermissionError("injected publication failure")

    monkeypatch.setattr(Path, "replace", cannot_replace)
    _write_generation(9.0)

    assert _scores(frames._read_disk_cache(REPO, None)) == (1.0, 1.25, 1.5)
    assert list(cache_dir.glob("*.tmp")) == []


def test_archive_close_failure_keeps_previous_complete_snapshot(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    original = zipfile.ZipFile.close

    def fail_writer_close(self):
        was_writing = self.mode == "w" and self.fp is not None
        original(self)
        if was_writing:
            raise OSError("injected archive close failure")

    monkeypatch.setattr(zipfile.ZipFile, "close", fail_writer_close)
    _write_generation(9.0)

    assert _scores(frames._read_disk_cache(REPO, None)) == (1.0, 1.25, 1.5)
    assert list(cache_dir.glob("*.tmp")) == []


def test_reader_between_normal_publications_never_sees_mixed_frames(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    original = Path.replace
    observed = []

    def read_after_first_replace(self, target):
        result = original(self, target)
        if not observed:
            observed.append(frames._read_disk_cache(REPO, None))
        return result

    monkeypatch.setattr(Path, "replace", read_after_first_replace)
    _write_generation(9.0)

    assert len(observed) == 1
    assert _scores(observed[0]) in {(1.0, 1.25, 1.5), (9.0, 9.25, 9.5)}
    assert _scores(frames._read_disk_cache(REPO, None)) == (9.0, 9.25, 9.5)


def test_reader_open_before_replace_keeps_one_complete_generation(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    original = pl.read_parquet
    replaced = False

    def replace_after_first_frame(*args: Any, **kwargs: Any):
        nonlocal replaced
        result = original(*args, **kwargs)
        if not replaced:
            replaced = True
            _write_generation(9.0)
        return result

    monkeypatch.setattr(pl, "read_parquet", replace_after_first_frame)
    result = frames._read_disk_cache(REPO, None)

    # Windows may refuse the writer's replacement while the reader is open.
    assert result is None or _scores(result) == (1.0, 1.25, 1.5)
    assert _scores(frames._read_disk_cache(REPO, None)) in {
        (1.0, 1.25, 1.5),
        (9.0, 9.25, 9.5),
    }


def test_deleted_owner_cannot_return_from_undeletable_legacy_shard(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    legacy = cache_dir / "second.parquet"
    _frame(1.25).write_parquet(legacy)
    original = Path.unlink

    def cannot_remove_old_owner(self, *args: Any, **kwargs: Any):
        if self == legacy:
            raise PermissionError("injected obsolete-shard cleanup failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cannot_remove_old_owner)
    _write_generation(9.0, second_empty=True)

    assert _scores(frames._read_disk_cache(REPO, "sha-9")) == (9.0, None, 9.5)


def test_writing_snapshot_preserves_shared_and_legacy_files(cache_dir: Path):
    shared = cache_dir / "benchmark_results.parquet"
    legacy = cache_dir / "old-owner.parquet"
    _frame(7.0).write_parquet(shared)
    _frame(3.0).write_parquet(legacy)
    old_manifest = cache_dir / "manifest.json"
    old_manifest.write_text('{"schema_version": 3}', encoding="utf-8")
    before = {p: p.read_bytes() for p in [shared, legacy, old_manifest]}

    _write_generation(9.0)

    assert {p: p.read_bytes() for p in before} == before
    assert _scores(frames._read_disk_cache(REPO, None)) == (9.0, 9.25, 9.5)


def test_two_writers_use_unique_temporary_files_and_publish_complete_snapshots(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_generation(1.0)
    barrier = threading.Barrier(2)
    local = threading.local()
    lock = threading.Lock()
    sources = []
    original_write = pl.DataFrame.write_parquet
    original_replace = Path.replace

    def overlap_first_frame(self, *args: Any, **kwargs: Any):
        result = original_write(self, *args, **kwargs)
        if not getattr(local, "first_frame_written", False):
            local.first_frame_written = True
            barrier.wait(timeout=10)
        return result

    def record_replace(self, target):
        with lock:
            sources.append(self)
        return original_replace(self, target)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", overlap_first_frame)
    monkeypatch.setattr(Path, "replace", record_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_write_generation, value) for value in [3.0, 9.0]]
        for future in futures:
            future.result(timeout=15)

    assert len(sources) == len(set(sources)) == 2
    assert _scores(frames._read_disk_cache(REPO, None)) in {
        (3.0, 3.25, 3.5),
        (9.0, 9.25, 9.5),
    }
    assert list(cache_dir.glob("*.tmp")) == []


def _parquet_bytes(score: float) -> bytes:
    buffer = io.BytesIO()
    _frame(score).write_parquet(buffer)
    return buffer.getvalue()


def _fixture_archive(cache_dir: Path, corruption: str | None = None) -> Path:
    path = cache_dir / SNAPSHOT_FILE
    manifest = {
        "repo_id": REPO,
        "sha": "fixture-sha",
        "schema_version": 4,
        "benchmarks": {"first": "benchmarks/first.parquet"},
    }
    entries = {
        "unified.parquet": _parquet_bytes(7.5),
        "benchmarks/first.parquet": _parquet_bytes(7.0),
    }
    if corruption in {"schema", "repo", "sha"}:
        key = {"schema": "schema_version", "repo": "repo_id", "sha": "sha"}[corruption]
        manifest[key] = 3 if corruption == "schema" else "different"
    elif corruption == "missing_sha":
        del manifest["sha"]
    elif corruption == "bad_owner_type":
        manifest["benchmarks"] = ["first"]
    elif corruption == "wrong_entry_namespace":
        manifest["benchmarks"] = {"first": "../first.parquet"}
    elif corruption == "missing_unified":
        del entries["unified.parquet"]
    elif corruption == "missing_owner":
        del entries["benchmarks/first.parquet"]
    elif corruption == "bad_parquet":
        entries["benchmarks/first.parquet"] = b"not a parquet file"
    if corruption == "manifest_type":
        manifest_bytes = b"[]"
    elif corruption == "bad_json":
        manifest_bytes = b"{"
    elif corruption == "duplicate_owner_key":
        manifest_bytes = (
            b'{"repo_id":"test/results","sha":"fixture-sha",'
            b'"schema_version":4,"benchmarks":{'
            b'"first":"benchmarks/first.parquet",'
            b'"first":"benchmarks/first.parquet"}}'
        )
    else:
        manifest_bytes = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        if corruption != "missing_manifest":
            archive.writestr("manifest.json", manifest_bytes)
        for name, data in entries.items():
            archive.writestr(name, data)
        if corruption == "duplicate_zip_entry":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("benchmarks/first.parquet", _parquet_bytes(9.0))
    if corruption == "bad_zip":
        path.write_bytes(path.read_bytes()[:40])
    elif corruption == "crc":
        with zipfile.ZipFile(path) as archive:
            offset = archive.getinfo("unified.parquet").header_offset
        data = bytearray(path.read_bytes())
        name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
        data[offset + 30 + name_length + extra_length] ^= 1
        path.write_bytes(data)
    elif corruption == "unsupported_version":
        data = bytearray(path.read_bytes())
        central_header = data.index(b"PK\x01\x02")
        struct.pack_into("<H", data, central_header + 6, 99)
        path.write_bytes(data)
    return path


def test_reader_uses_explicit_archive_entries_not_root_parquets(cache_dir: Path):
    _fixture_archive(cache_dir)
    _frame(99.0).write_parquet(cache_dir / "second.parquet")
    _frame(98.0).write_parquet(cache_dir / "benchmark_results.parquet")

    assert _scores(frames._read_disk_cache(REPO, "fixture-sha")) == (7.0, None, 7.5)


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "repo",
        "sha",
        "manifest_type",
        "missing_manifest",
        "bad_json",
        "missing_sha",
        "bad_owner_type",
        "wrong_entry_namespace",
        "duplicate_owner_key",
        "duplicate_zip_entry",
        "missing_unified",
        "missing_owner",
        "bad_parquet",
        "bad_zip",
        "crc",
        "unsupported_version",
    ],
)
def test_invalid_snapshot_is_a_complete_miss(cache_dir: Path, corruption: str):
    _fixture_archive(cache_dir, corruption)

    assert frames._read_disk_cache(REPO, "fixture-sha") is None


def test_loader_rebuilds_snapshot_with_unsupported_zip_version(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _fixture_archive(cache_dir, "unsupported_version")
    fresh = _frame(9.0)
    hub_calls = []

    def load_fresh(repo_id):
        hub_calls.append(repo_id)
        return fresh

    monkeypatch.setattr(
        frames, "get_settings", lambda: Settings(cache_repo=REPO, disk_cache=True)
    )
    monkeypatch.setattr(frames, "_hf_dataset_sha", lambda repo_id: "fixture-sha")
    monkeypatch.setattr(frames, "get_cache", object)
    monkeypatch.setattr(frames, "_load_default_from_hub", load_fresh)
    monkeypatch.setattr(frames.mteb, "get_benchmarks", list)
    frames._load_per_benchmark_frames.cache_clear()
    try:
        loaded, unified = frames._load_per_benchmark_frames()

        assert hub_calls == [REPO]
        assert loaded == {}
        assert_frame_equal(unified, fresh)
        restored = frames._read_disk_cache(REPO, "fixture-sha")
        assert restored is not None
        assert restored[0] == {}
        assert_frame_equal(restored[1], fresh)
    finally:
        frames._load_per_benchmark_frames.cache_clear()


def test_legacy_schema_three_is_ignored(cache_dir: Path):
    (cache_dir / "manifest.json").write_text(
        json.dumps({"repo_id": REPO, "sha": "old", "schema_version": 3}),
        encoding="utf-8",
    )
    _frame(1.0).write_parquet(cache_dir / "_unified.parquet")
    _frame(1.0).write_parquet(cache_dir / "first.parquet")

    assert frames._read_disk_cache(REPO, None) is None


def test_snapshot_supports_zip64_entries(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # Lower the real ZIP limit so small real parquet payloads require ZIP64.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 128)
    _write_generation(9.0)

    assert _scores(frames._read_disk_cache(REPO, None)) == (9.0, 9.25, 9.5)
    with zipfile.ZipFile(cache_dir / SNAPSHOT_FILE) as archive:
        assert archive.testzip() is None
