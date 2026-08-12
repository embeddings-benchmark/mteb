"""Loader invariants for InsAVE80KVT2VRetrieval.

These run against small synthetic archives rather than the 1.55 GB upstream shard, so
they exercise the fail-loudly guards without touching the network.
"""

import csv
import io
import tarfile
from pathlib import Path

import pytest

from mteb.tasks.retrieval.eng.insave_vt2v_retrieval import (
    _archive_media_members,
    _read_rows,
    _resolve_corpus_files,
)


def _make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_flat_archive_members_are_mapped_to_basenames(tmp_path: Path) -> None:
    tar_path = tmp_path / "flat.tar"
    _make_tar(tar_path, {"eval/00001.mp4": b"a", "eval/00002.mp4": b"b"})

    assert _archive_media_members(tar_path) == {
        "00001.mp4": "eval/00001.mp4",
        "00002.mp4": "eval/00002.mp4",
    }


def test_nested_duplicate_basenames_cannot_silently_collapse(tmp_path: Path) -> None:
    """`a/00001.mp4` and `b/00001.mp4` must fail loudly, not overwrite each other."""
    tar_path = tmp_path / "nested.tar"
    _make_tar(tar_path, {"a/00001.mp4": b"first", "b/00001.mp4": b"second"})

    with pytest.raises(ValueError, match="share the basename"):
        _archive_media_members(tar_path)


def test_archive_without_media_members_fails(tmp_path: Path) -> None:
    tar_path = tmp_path / "empty.tar"
    _make_tar(tar_path, {})

    with pytest.raises(ValueError, match="no media members"):
        _archive_media_members(tar_path)


def test_document_ids_claimed_by_two_clips_fail(tmp_path: Path) -> None:
    """Differently-nested clips sharing a path stem must not collapse into one id."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "00001.mp4").write_bytes(b"a")
    (video_dir / "00002.mp4").write_bytes(b"b")

    rows = [
        {
            "original_video": "a/00001.mp4",
            "target_video": "00002.mp4",
            "instruction": "edit it",
        },
        {
            "original_video": "b/00001.mp4",
            "target_video": "00002.mp4",
            "instruction": "edit it differently",
        },
    ]

    with pytest.raises(ValueError, match="claimed by two different clips"):
        _resolve_corpus_files(rows, video_dir)


@pytest.mark.parametrize("missing", ["original_video", "target_video", "instruction"])
def test_missing_required_column_fails(tmp_path: Path, missing: str) -> None:
    columns = [
        c for c in ("original_video", "target_video", "instruction") if c != missing
    ]
    csv_path = tmp_path / "eval.csv"
    _write_csv(csv_path, [dict.fromkeys(columns, "x")], columns)

    with pytest.raises(ValueError, match="missing required column"):
        _read_rows(csv_path)


def test_unused_and_extra_columns_are_permitted(tmp_path: Path) -> None:
    """`instruction_reverse` is unused, and unknown upstream columns are ignored."""
    columns = ["original_video", "target_video", "instruction", "some_future_column"]
    csv_path = tmp_path / "eval.csv"
    _write_csv(
        csv_path,
        [
            {
                "original_video": "eval/00001_original.mp4",
                "target_video": "eval/00001_traget.mp4",
                "instruction": "edit it",
                "some_future_column": "ignored",
            }
        ],
        columns,
    )

    rows = _read_rows(csv_path)

    assert len(rows) == 1
    assert rows[0]["instruction"] == "edit it"
    assert "instruction_reverse" not in rows[0]
