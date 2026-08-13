from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[4] / "scripts" / "data" / "evve" / "create_data.py"
SPEC = importlib.util.spec_from_file_location("evve_create_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_manifest_is_pinned_and_well_formed() -> None:
    video_ids = MODULE.load_protocol_ids()
    s2vs_video_ids = MODULE.load_protocol_ids(MODULE.S2VS_PROTOCOL_MANIFEST)
    source_overrides = MODULE.load_media_source_overrides()

    assert len(video_ids) == 2_110
    assert len(s2vs_video_ids) == 2_410
    assert set(video_ids) < set(s2vs_video_ids)
    assert len(set(video_ids)) == len(video_ids)
    assert video_ids == tuple(sorted(video_ids))
    assert source_overrides["-ZoR3OTbPx4"]["sha256"] == (
        "2921fb9952a009dc34e1d67c341c70f5de6aa82aaf5c77a2e03b6e889e228c43"
    )
    assert source_overrides["8bTYodeSLlU"]["sha256"] == (
        "bcf2497331719d1a9513aed16d1a03ba2f48fad03473b79387c797d49fbc1d69"
    )


def test_build_protocol_is_deterministic() -> None:
    annotations = {
        "queries": {"query00001", "query00002"},
        "database": {"corpus00001", "corpus00002", "corpus00003"},
        "annotation": {
            "event-b": ({"query00002"}, {"corpus00003"}, set()),
            "event-a": (
                {"query00001"},
                {"corpus00002", "corpus00001"},
                set(),
            ),
        },
    }
    protocol_ids = (
        "corpus00001",
        "corpus00002",
        "corpus00003",
        "query00001",
        "query00002",
    )

    protocol = MODULE.build_protocol(
        annotations, protocol_ids, enforce_expected_counts=False
    )

    assert protocol.queries == ("query00001", "query00002")
    assert protocol.database == (
        "corpus00001",
        "corpus00002",
        "corpus00003",
    )
    assert protocol.qrels == (
        ("query00001", "corpus00001", 1),
        ("query00001", "corpus00002", 1),
        ("query00002", "corpus00003", 1),
    )
    assert protocol.query_events == {
        "query00001": "event-a",
        "query00002": "event-b",
    }
    assert protocol.query_ignored == {
        "query00001": (),
        "query00002": (),
    }
    assert {query_id for query_id, _, score in protocol.qrels if score > 0} == set(
        protocol.queries
    )
    assert [row["event"] for row in protocol.event_stats] == ["event-a", "event-b"]
    assert MODULE.protocol_summary(protocol)["published_original"]["qrels"] == 3
    assert (
        MODULE.protocol_summary(protocol)["evaluation_protocol"][
            "queries_without_positives"
        ]
        == 0
    )


def test_protocol_summary_records_event_coverage_before_and_after() -> None:
    annotations = {
        "queries": {"query00001", "query00002"},
        "database": {"corpus00001", "corpus00002"},
        "annotation": {
            "event": (
                {"query00001", "query00002"},
                {"corpus00001", "corpus00002"},
                set(),
            ),
        },
    }
    before_filter = MODULE.build_protocol(
        annotations,
        ("corpus00001", "corpus00002", "query00001", "query00002"),
        enforce_expected_counts=False,
    )
    protocol = MODULE.build_protocol(
        annotations,
        ("corpus00001", "query00001"),
        enforce_expected_counts=False,
    )

    summary = MODULE.protocol_summary(protocol, before_filter)

    assert summary["events"] == [
        {
            "event": "event",
            "queries": 1,
            "positives": 1,
            "null": 0,
            "original_queries": 2,
            "original_positives": 2,
            "before_filter_queries": 2,
            "before_filter_positives": 2,
            "removed_queries": 1,
            "removed_positives": 1,
        }
    ]


def test_build_protocol_rejects_unknown_ids() -> None:
    annotations = {
        "queries": {"query00001"},
        "database": {"corpus00001"},
        "annotation": {
            "event": ({"query00001"}, {"corpus00001"}, set()),
        },
    }

    with pytest.raises(ValueError, match="unknown IDs"):
        MODULE.build_protocol(
            annotations,
            ("unknown0001",),
            enforce_expected_counts=False,
        )


def test_download_id_file_must_be_a_unique_protocol_subset(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    path.write_text("video000001\nvideo000002\n", encoding="utf-8")

    assert MODULE.load_download_ids(
        path, ("video000001", "video000002", "video000003")
    ) == ("video000001", "video000002")

    path.write_text("video000001\nvideo000001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        MODULE.load_download_ids(path, ("video000001",))

    path.write_text("outside0001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the frozen protocol"):
        MODULE.load_download_ids(path, ("video000001",))


def test_media_audit_classifies_unavailable_protocol_items(tmp_path: Path) -> None:
    protocol = MODULE.Protocol(
        queries=("query00001", "query00002"),
        database=("corpus00001", "corpus00002", "corpus00003"),
        qrels=(
            ("query00001", "corpus00001", 1),
            ("query00001", "corpus00002", 1),
            ("query00002", "corpus00003", 1),
        ),
        query_events={"query00001": "event-a", "query00002": "event-b"},
        query_ignored={"query00001": (), "query00002": ()},
        event_stats=(),
    )
    present = {
        "query00001": tmp_path / "query00001.mp4",
        "corpus00001": tmp_path / "corpus00001.mp4",
        "corpus00002": tmp_path / "corpus00002.mp4",
    }

    audit = MODULE.media_audit(protocol, present, {"corpus00002"})

    assert audit == {
        "required": 5,
        "present_and_decodable": 2,
        "unavailable": 3,
        "unavailable_queries": 1,
        "unavailable_database": 2,
        "unavailable_positive_database": 2,
        "unavailable_other_database": 0,
        "unavailable_qrels": 2,
        "invalid_media": 1,
        "events": [
            {
                "event": "event-a",
                "unavailable_queries": 0,
                "unavailable_positives": 1,
            },
            {
                "event": "event-b",
                "unavailable_queries": 1,
                "unavailable_positives": 1,
            },
        ],
    }


def test_media_index_accepts_flat_and_nested_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "flatvideo01.mp4"
    flat.write_bytes(b"video")
    nested = tmp_path / "nestedvid01" / "video.webm"
    nested.parent.mkdir()
    nested.write_bytes(b"video")

    media = MODULE.index_media(tmp_path)

    assert media == {
        "flatvideo01": flat.resolve(),
        "nestedvid01": nested.resolve(),
    }


def test_media_index_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = tmp_path / "duplicate01.mp4"
    first.write_bytes(b"video")
    second = tmp_path / "duplicate01" / "video.webm"
    second.parent.mkdir()
    second.write_bytes(b"video")

    with pytest.raises(ValueError, match="multiple media files"):
        MODULE.index_media(tmp_path)


@pytest.mark.parametrize(
    ("probe_stdout", "expected"),
    [
        ('{"frames": [{"media_type": "video"}]}', True),
        ('{"frames": []}', False),
        ("{}", False),
    ],
)
def test_video_validation_requires_decodable_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_stdout: str,
    expected: bool,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x" * 1_025)
    monkeypatch.setattr(MODULE.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_, **__: MODULE.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=probe_stdout,
            stderr="",
        ),
    )

    assert MODULE._video_is_decodable(video) is expected


def test_download_failure_priority() -> None:
    assert MODULE._download_failure_priority(None) == 0
    assert MODULE._download_failure_priority("timed out") == 1
    assert MODULE._download_failure_priority("Sign in to confirm you’re not a bot") == 1
    assert MODULE._download_failure_priority("Private video") == 2
    assert MODULE._download_failure_priority("Video unavailable") == 2
    assert MODULE._download_failure_priority("Sign in to confirm your age") == 2


def test_download_failure_log_round_trip(tmp_path: Path) -> None:
    failure_log = tmp_path / "download-failures.jsonl"
    failure_log.write_text(
        '{"video_id": "video000001", "error": "Video unavailable"}\n'
        '{"video_id": "video000002", "error": "timed out"}\n',
        encoding="utf-8",
    )

    assert MODULE._read_download_failures(failure_log) == {
        "video000001": "Video unavailable",
        "video000002": "timed out",
    }


def test_download_resume_prioritizes_untried_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "existing001.mp4").write_bytes(b"existing")
    failure_log = tmp_path / "download-failures.jsonl"
    failure_log.write_text(
        '{"video_id": "transient01", "error": "timed out"}\n'
        '{"video_id": "hardfail001", "error": "Private video"}\n',
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_download(video_id: str, media_root: Path, **_: object) -> Path:
        calls.append(video_id)
        destination = media_root / f"{video_id}.mp4"
        destination.write_bytes(b"downloaded")
        return destination

    monkeypatch.setattr(MODULE, "_resolve_yt_dlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(MODULE, "_video_is_decodable", lambda _: True)
    monkeypatch.setattr(MODULE, "_download_video", fake_download)

    outcomes = MODULE.download_media(
        ("existing001", "hardfail001", "transient01", "untried0001"),
        media_dir,
        workers=1,
        cookies_from_browser=None,
        retries=0,
        concurrent_fragments=1,
        sleep_min=0,
        sleep_max=0,
        extra_args=(),
        source_overrides={},
        limit=None,
        failure_log=failure_log,
    )

    assert calls == ["untried0001", "transient01", "hardfail001"]
    assert set(outcomes) == {
        "existing001",
        "hardfail001",
        "transient01",
        "untried0001",
    }
    assert not failure_log.read_text(encoding="utf-8")


def test_dataset_schema_preserves_empty_ignored_id_lists(tmp_path: Path) -> None:
    from datasets import Sequence, Value

    query = tmp_path / "query00001.mp4"
    query.write_bytes(b"placeholder")
    corpus = tmp_path / "corpus00001.mp4"
    corpus.write_bytes(b"placeholder")
    protocol = MODULE.Protocol(
        queries=("query00001",),
        database=("corpus00001",),
        qrels=(("query00001", "corpus00001", 1),),
        query_events={"query00001": "event"},
        query_ignored={"query00001": ()},
        event_stats=(),
    )

    datasets = MODULE.build_datasets(
        protocol,
        {"query00001": query, "corpus00001": corpus},
    )

    assert datasets["queries"].features["ignored_corpus_ids"] == Sequence(
        Value("string")
    )
    assert datasets["qrels"].features["score"] == Value("int32")
    assert datasets["queries"]["media_source_url"][0].endswith("query00001")
