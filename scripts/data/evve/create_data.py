#!/usr/bin/env python3
"""Construct and package a checksum-pinned surviving-public-media EVVE set.

The official EVVE metadata identifies media by YouTube ID. This script freezes
the reproducibly obtainable 2,110-video subset of the 2,410 IDs retained by the
public S2VS feature artifact, downloads a compact progressive representation,
validates every retained item, and builds the MTEB corpus / queries / qrels
configurations. Downloads are resumable and failures never silently change the
benchmark.

Examples:
    uv run python scripts/data/evve/create_data.py \
        --work-dir /tmp/evve-mteb

    uv run python scripts/data/evve/create_data.py \
        --work-dir /tmp/evve-mteb \
        --download --workers 8

    uv run python scripts/data/evve/create_data.py \
        --work-dir /tmp/evve-mteb \
        --repo-id Cerru02/EVVE --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATIONS_REVISION = "03eb0c9a9a7455d132c4210b797fa6ef563b52ea"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/gkordo/s2vs/"
    f"{ANNOTATIONS_REVISION}/data/evve.pickle"
)
ANNOTATIONS_SHA256 = "0898fafc817b9457be25cafeefa996d78d1021fb4816a9e3c1a14b7493023410"
FEATURES_URL = "https://mever.iti.gr/s2vs/features/evve.hdf5"
EVALUATOR_REVISION = "c0f59d49ffe6b4069809794697554c8a1ce969d9"
EVALUATOR_URL = (
    f"https://raw.githubusercontent.com/fyang93/BURST/{EVALUATOR_REVISION}/eval_evve.py"
)
EVALUATOR_SHA256 = "49dd29a499857764972b6b8959ad18dd94d80f8b34e8c4814c05076609fd93fd"
S2VS_PROTOCOL_MANIFEST = Path(__file__).with_name("s2vs-2023-video-ids.txt")
S2VS_PROTOCOL_MANIFEST_SHA256 = (
    "d7a84d997fcc23c6bb10db1902203b5c0d6ad65c65c4f8257b4295c690fb1c1c"
)
PROTOCOL_MANIFEST = Path(__file__).with_name("surviving-public-media-video-ids.txt")
MEDIA_SOURCE_OVERRIDES = Path(__file__).with_name("media-source-overrides.json")
PROTOCOL_MANIFEST_SHA256 = (
    "ebf63bd9f524a1ab15104657e0448ddd4a1cbfb5720795ca9569703fbffcb0e6"
)

EXPECTED_QUERIES = 466
EXPECTED_DATABASE = 1_644
EXPECTED_EVENTS = 13
EXPECTED_QRELS = 86_925
S2VS_QUERIES = 504
S2VS_DATABASE = 1_906
S2VS_QRELS = 100_789
PUBLISHED_ORIGINAL_QUERIES = 620
PUBLISHED_ORIGINAL_DATABASE = 2_375

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_VIDEO_EXTENSIONS = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"}
_YOUTUBE_URL = "https://www.youtube.com/watch?v={video_id}"
_DOWNLOAD_FORMAT = "b[height<=360][ext=mp4]/b[height<=360]/b[ext=mp4]/b"
_DOWNLOAD_FORMAT_OVERRIDES = {
    # YouTube's progressive format for these IDs currently contains only MP4
    # headers. Separate video/audio streams contain the complete media payload.
    video_id: "bv[height<=360][ext=mp4]+ba[ext=m4a]"
    for video_id in ("CikHK-UfUxc", "TWMgceoDuA0", "Ua4I9ccQYJ0", "ntHfmfOk8Jw")
}


@dataclass(frozen=True)
class Protocol:
    queries: tuple[str, ...]
    database: tuple[str, ...]
    qrels: tuple[tuple[str, str, int], ...]
    query_events: dict[str, str]
    query_ignored: dict[str, tuple[str, ...]]
    event_stats: tuple[dict[str, int | str], ...]


class DownloadFailure(Exception):
    """A non-fatal media download failure collected in the audit log."""


class _RestrictedUnpickler(pickle.Unpickler):
    """Load the pinned legacy annotation file without arbitrary globals."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name == "set":
            return set
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_annotations(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    digest = _sha256(data)
    if digest != ANNOTATIONS_SHA256:
        raise ValueError(
            f"annotation checksum mismatch for {path}: {digest}; "
            f"expected {ANNOTATIONS_SHA256}"
        )
    payload = _RestrictedUnpickler(io.BytesIO(data)).load()
    if not isinstance(payload, dict) or set(payload) != {
        "annotation",
        "database",
        "queries",
    }:
        raise ValueError("unexpected EVVE annotation structure")
    return payload


def fetch_annotations(work_dir: Path) -> Path:
    destination = work_dir / "source" / "evve.pickle"
    if destination.is_file():
        load_annotations(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ANNOTATIONS_URL) as response:
        data = response.read()
    if _sha256(data) != ANNOTATIONS_SHA256:
        raise ValueError("downloaded EVVE annotations failed checksum validation")

    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, delete=False
    ) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    os.replace(temporary, destination)
    return destination


def load_protocol_ids(path: Path = PROTOCOL_MANIFEST) -> tuple[str, ...]:
    data = path.read_bytes()
    digest = _sha256(data)
    expected_digest = {
        PROTOCOL_MANIFEST: PROTOCOL_MANIFEST_SHA256,
        S2VS_PROTOCOL_MANIFEST: S2VS_PROTOCOL_MANIFEST_SHA256,
    }.get(path)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            f"protocol manifest checksum mismatch: {digest}; expected {expected_digest}"
        )

    video_ids = tuple(line for line in data.decode().splitlines() if line)
    if video_ids != tuple(sorted(video_ids)):
        raise ValueError("protocol manifest must be sorted")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("protocol manifest contains duplicate video IDs")
    invalid = [video_id for video_id in video_ids if not _VIDEO_ID.fullmatch(video_id)]
    if invalid:
        raise ValueError(f"invalid video IDs in protocol manifest: {invalid[:5]}")
    return video_ids


def load_download_ids(path: Path, protocol_ids: tuple[str, ...]) -> tuple[str, ...]:
    video_ids = tuple(
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    )
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(f"download ID file contains duplicates: {path}")
    invalid = [video_id for video_id in video_ids if not _VIDEO_ID.fullmatch(video_id)]
    if invalid:
        raise ValueError(f"invalid download IDs in {path}: {invalid[:5]}")
    outside_protocol = sorted(set(video_ids) - set(protocol_ids))
    if outside_protocol:
        raise ValueError(
            f"download IDs are outside the frozen protocol: {outside_protocol[:5]}"
        )
    return video_ids


def load_media_source_overrides(
    path: Path = MEDIA_SOURCE_OVERRIDES,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("media source overrides must be a JSON object")

    protocol_ids = set(load_protocol_ids())
    overrides: dict[str, dict[str, str]] = {}
    for video_id, raw_record in payload.items():
        if video_id not in protocol_ids or not isinstance(raw_record, dict):
            raise ValueError(f"invalid media source override for {video_id!r}")
        record = {str(key): str(value) for key, value in raw_record.items()}
        if set(record) != {"extension", "sha256", "url"}:
            raise ValueError(f"unexpected media source fields for {video_id!r}")
        if record["extension"] not in _VIDEO_EXTENSIONS:
            raise ValueError(f"invalid media extension for {video_id!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"invalid media checksum for {video_id!r}")
        if not record["url"].startswith("https://"):
            raise ValueError(f"media source must use HTTPS for {video_id!r}")
        overrides[video_id] = record
    return overrides


def build_protocol(
    annotations: dict[str, Any],
    protocol_ids: tuple[str, ...],
    *,
    enforce_expected_counts: bool = True,
) -> Protocol:
    available = set(protocol_ids)
    all_queries = set(annotations["queries"])
    all_database = set(annotations["database"])
    unknown = available - all_queries - all_database
    if unknown:
        raise ValueError(f"protocol contains unknown IDs: {sorted(unknown)[:5]}")

    queries = tuple(sorted(available & all_queries))
    database = tuple(sorted(available & all_database))
    if set(queries) & set(database):
        raise ValueError("EVVE query and database splits overlap")

    qrels: list[tuple[str, str, int]] = []
    query_events: dict[str, str] = {}
    query_ignored: dict[str, tuple[str, ...]] = {}
    event_stats: list[dict[str, int | str]] = []
    annotated_queries: set[str] = set()

    events = annotations["annotation"]
    if not isinstance(events, dict):
        raise ValueError("EVVE annotation events must be a mapping")

    for event, annotation in sorted(events.items()):
        if not isinstance(annotation, tuple) or len(annotation) != 3:
            raise ValueError(f"unexpected annotation for event {event!r}")
        event_queries, event_positives, event_null = map(set, annotation)
        if event_queries & all_database:
            raise ValueError(f"event {event!r} places queries in the database")
        if not event_positives <= all_database:
            raise ValueError(f"event {event!r} has positives outside the database")

        kept_queries = tuple(sorted(event_queries & available))
        kept_positives = tuple(sorted(event_positives & available))
        kept_null = tuple(sorted(event_null & available))
        if not kept_queries or not kept_positives:
            raise ValueError(f"event {event!r} has an empty reduced split")

        for query_id in kept_queries:
            if query_id in query_events:
                raise ValueError(f"query {query_id!r} belongs to multiple events")
            query_events[query_id] = event
            query_ignored[query_id] = kept_null
            annotated_queries.add(query_id)
            qrels.extend((query_id, corpus_id, 1) for corpus_id in kept_positives)

        event_stats.append(
            {
                "event": event,
                "queries": len(kept_queries),
                "positives": len(kept_positives),
                "null": len(kept_null),
                "original_queries": len(event_queries),
                "original_positives": len(event_positives),
            }
        )

    if annotated_queries != set(queries):
        missing = set(queries) - annotated_queries
        raise ValueError(f"queries without an event annotation: {sorted(missing)[:5]}")

    queries_with_positives = {query_id for query_id, _, score in qrels if score > 0}
    if queries_with_positives != set(queries):
        missing = set(queries) - queries_with_positives
        raise ValueError(f"queries without a positive qrel: {sorted(missing)[:5]}")

    protocol = Protocol(
        queries=queries,
        database=database,
        qrels=tuple(qrels),
        query_events=query_events,
        query_ignored=query_ignored,
        event_stats=tuple(event_stats),
    )
    if enforce_expected_counts:
        actual = (
            len(protocol.queries),
            len(protocol.database),
            len(protocol.event_stats),
            len(protocol.qrels),
        )
        expected = (
            EXPECTED_QUERIES,
            EXPECTED_DATABASE,
            EXPECTED_EVENTS,
            EXPECTED_QRELS,
        )
        if actual != expected:
            raise ValueError(
                f"reduced protocol counts changed: {actual}; expected {expected}"
            )
    return protocol


def _protocol_counts(protocol: Protocol) -> dict[str, int]:
    positive_ids = {corpus_id for _, corpus_id, score in protocol.qrels if score > 0}
    queries_with_positives = {
        query_id for query_id, _, score in protocol.qrels if score > 0
    }
    return {
        "queries": len(protocol.queries),
        "database": len(protocol.database),
        "events": len(protocol.event_stats),
        "qrels": len(protocol.qrels),
        "positive_database_videos": len(positive_ids),
        "other_database_videos": len(protocol.database) - len(positive_ids),
        "query_database_overlap": len(set(protocol.queries) & set(protocol.database)),
        "queries_without_positives": len(
            set(protocol.queries) - queries_with_positives
        ),
    }


def protocol_summary(
    protocol: Protocol, before_filter: Protocol | None = None
) -> dict[str, Any]:
    before_filter = before_filter or protocol
    original_qrels = 0
    original_positive_database = 0
    for event in protocol.event_stats:
        original_queries = event["original_queries"]
        original_positives = event["original_positives"]
        if not isinstance(original_queries, int) or not isinstance(
            original_positives, int
        ):
            raise TypeError("original event counts must be integers")
        original_qrels += original_queries * original_positives
        original_positive_database += original_positives
    before_events = {row["event"]: row for row in before_filter.event_stats}
    if set(before_events) != {row["event"] for row in protocol.event_stats}:
        raise ValueError("event coverage changed while filtering the protocol")
    event_coverage = []
    for row in protocol.event_stats:
        before_row = before_events[row["event"]]
        integer_fields = (
            before_row["queries"],
            before_row["positives"],
            row["queries"],
            row["positives"],
        )
        if not all(isinstance(value, int) for value in integer_fields):
            raise TypeError("event coverage counts must be integers")
        before_queries, before_positives, queries, positives = integer_fields
        assert isinstance(before_queries, int)
        assert isinstance(before_positives, int)
        assert isinstance(queries, int)
        assert isinstance(positives, int)
        event_coverage.append(
            {
                **row,
                "before_filter_queries": before_queries,
                "before_filter_positives": before_positives,
                "removed_queries": before_queries - queries,
                "removed_positives": before_positives - positives,
            }
        )

    return {
        "schema_version": 2,
        "protocol": "evve-surviving-public-media-2026-08-11",
        "sources": {
            "annotations_revision": ANNOTATIONS_REVISION,
            "annotations_url": ANNOTATIONS_URL,
            "annotations_sha256": ANNOTATIONS_SHA256,
            "feature_artifact_url": FEATURES_URL,
            "surviving_evaluator_revision": EVALUATOR_REVISION,
            "surviving_evaluator_url": EVALUATOR_URL,
            "surviving_evaluator_sha256": EVALUATOR_SHA256,
            "s2vs_protocol_manifest": S2VS_PROTOCOL_MANIFEST.name,
            "s2vs_protocol_manifest_sha256": S2VS_PROTOCOL_MANIFEST_SHA256,
            "protocol_manifest": PROTOCOL_MANIFEST.name,
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        },
        "published_original": {
            "queries": PUBLISHED_ORIGINAL_QUERIES,
            "database": PUBLISHED_ORIGINAL_DATABASE,
            "events": EXPECTED_EVENTS,
            "qrels": original_qrels,
            "positive_database_videos": original_positive_database,
            "other_database_videos": (
                PUBLISHED_ORIGINAL_DATABASE - original_positive_database
            ),
        },
        "before_filter_s2vs_protocol": _protocol_counts(before_filter),
        "evaluation_protocol": _protocol_counts(protocol),
        "events": event_coverage,
    }


def media_audit(
    protocol: Protocol,
    media: dict[str, Path],
    invalid: set[str] | None = None,
) -> dict[str, Any]:
    invalid = invalid or set()
    unavailable = (set(protocol.queries) | set(protocol.database)) - set(media)
    unavailable |= invalid
    unavailable_queries = set(protocol.queries) & unavailable
    unavailable_database = set(protocol.database) & unavailable
    positive_database = {
        corpus_id for _, corpus_id, score in protocol.qrels if score > 0
    }

    event_rows: list[dict[str, int | str]] = []
    for event in sorted(set(protocol.query_events.values())):
        event_queries = {
            query_id
            for query_id, query_event in protocol.query_events.items()
            if query_event == event
        }
        event_positives = {
            corpus_id
            for query_id, corpus_id, score in protocol.qrels
            if protocol.query_events[query_id] == event and score > 0
        }
        event_rows.append(
            {
                "event": event,
                "unavailable_queries": len(event_queries & unavailable),
                "unavailable_positives": len(event_positives & unavailable),
            }
        )

    return {
        "required": len(protocol.queries) + len(protocol.database),
        "present_and_decodable": len(protocol.queries)
        + len(protocol.database)
        - len(unavailable),
        "unavailable": len(unavailable),
        "unavailable_queries": len(unavailable_queries),
        "unavailable_database": len(unavailable_database),
        "unavailable_positive_database": len(unavailable_database & positive_database),
        "unavailable_other_database": len(unavailable_database - positive_database),
        "unavailable_qrels": sum(
            query_id in unavailable or corpus_id in unavailable
            for query_id, corpus_id, _ in protocol.qrels
        ),
        "invalid_media": len(invalid),
        "events": event_rows,
    }


def _resolve_yt_dlp() -> list[str]:
    try:
        import yt_dlp  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        binary = shutil.which("yt-dlp")
        if binary:
            probe = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, check=False
            )
            if probe.returncode == 0:
                return [binary]
        raise SystemExit(
            "yt-dlp is required for --download. Install it with `uv pip install yt-dlp`."
        ) from None
    return [sys.executable, "-m", "yt_dlp"]


def _video_is_decodable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1_024:
        return False
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                "%+#1",
                "-show_entries",
                "frame=media_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            return False
        try:
            frames = json.loads(probe.stdout)["frames"]
            return bool(frames and frames[0]["media_type"] == "video")
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return False

    try:
        from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]
    except ImportError as error:
        raise RuntimeError(
            "Media validation requires ffprobe or TorchCodec. Install FFmpeg or "
            "MTEB's `video` extra before downloading or packaging EVVE."
        ) from error
    try:
        duration = VideoDecoder(str(path)).metadata.duration_seconds
        return duration is not None and duration > 0
    except Exception:
        return False


def _media_candidates(media_dir: Path, video_id: str) -> list[Path]:
    return sorted(
        path
        for path in media_dir.glob(f"{video_id}.*")
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    )


def _download_fallback_media(
    video_id: str, media_dir: Path, source: dict[str, str]
) -> Path:
    destination = media_dir / f"{video_id}{source['extension']}"
    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    )
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile("wb", dir=media_dir, delete=False) as temporary,
        ):
            temporary_path = Path(temporary.name)
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                temporary.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != source["sha256"]:
            raise DownloadFailure(
                f"fallback checksum mismatch for {video_id}: {digest.hexdigest()}"
            )
        if temporary_path is None or not _video_is_decodable(temporary_path):
            raise DownloadFailure(f"fallback media is not decodable for {video_id}")
        os.replace(temporary_path, destination)
        return destination
    except DownloadFailure:
        raise
    except Exception as error:
        raise DownloadFailure(f"fallback download failed: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _download_video(
    video_id: str,
    media_dir: Path,
    *,
    yt_dlp: list[str],
    cookies_from_browser: str | None,
    retries: int,
    concurrent_fragments: int,
    sleep_min: float,
    sleep_max: float,
    extra_args: tuple[str, ...],
    fallback_source: dict[str, str] | None,
) -> Path:
    candidates = _media_candidates(media_dir, video_id)
    valid = [path for path in candidates if _video_is_decodable(path)]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise DownloadFailure(f"multiple valid media files found: {valid}")
    for path in candidates:
        path.unlink(missing_ok=True)

    media_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(media_dir / f"{video_id}.%(ext)s")
    last_error = ""
    for attempt in range(retries + 1):
        if attempt:
            delay = random.uniform(sleep_min, max(sleep_min, sleep_max))
            time.sleep(delay * 2 ** (attempt - 1))
        elif sleep_min > 0:
            time.sleep(random.uniform(sleep_min, max(sleep_min, sleep_max)))

        command = [
            *yt_dlp,
            "--format",
            _DOWNLOAD_FORMAT_OVERRIDES.get(video_id, _DOWNLOAD_FORMAT),
            "--no-playlist",
            "--no-mtime",
            "--no-warnings",
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "--concurrent-fragments",
            str(max(1, concurrent_fragments)),
            "--output",
            output_template,
            *extra_args,
        ]
        if cookies_from_browser:
            command.extend(["--cookies-from-browser", cookies_from_browser])
        command.append(_YOUTUBE_URL.format(video_id=video_id))

        process = subprocess.run(command, capture_output=True, text=True, check=False)
        candidates = _media_candidates(media_dir, video_id)
        valid = [path for path in candidates if _video_is_decodable(path)]
        if process.returncode == 0 and len(valid) == 1:
            return valid[0]
        last_error = (process.stderr or process.stdout or "yt-dlp failed").strip()
        for path in candidates:
            if path not in valid:
                path.unlink(missing_ok=True)

    if fallback_source is not None:
        return _download_fallback_media(video_id, media_dir, fallback_source)
    raise DownloadFailure(last_error[-1_000:] or "yt-dlp failed")


def _read_download_failures(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    failures: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
            video_id = row["video_id"]
            error = row["error"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid download failure at {path}:{line_number}"
            ) from exc
        if not isinstance(video_id, str) or not isinstance(error, str):
            raise ValueError(f"invalid download failure at {path}:{line_number}")
        failures[video_id] = error
    return failures


def _download_failure_priority(error: str | None) -> int:
    """Put untried IDs first, transient failures second, and hard failures last."""
    if error is None:
        return 0
    lowered = error.lower()
    hard_markers = (
        "private video",
        "video unavailable",
        "no longer available",
        "confirm your age",
    )
    return 2 if any(marker in lowered for marker in hard_markers) else 1


def _is_youtube_bot_gate(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "sign in to confirm you’re not a bot",
            "sign in to confirm you're not a bot",
        )
    )


def download_media(
    video_ids: tuple[str, ...],
    media_dir: Path,
    *,
    workers: int,
    cookies_from_browser: str | None,
    retries: int,
    concurrent_fragments: int,
    sleep_min: float,
    sleep_max: float,
    extra_args: tuple[str, ...],
    source_overrides: dict[str, dict[str, str]],
    limit: int | None,
    failure_log: Path,
) -> dict[str, Path]:
    selected = video_ids if limit is None else video_ids[:limit]
    yt_dlp = _resolve_yt_dlp()
    outcomes: dict[str, Path] = {}
    failures = _read_download_failures(failure_log)
    pending: list[str] = []

    # Validate existing media once before scheduling downloads. This keeps resumptions
    # cheap and prevents already-resolved IDs from consuming the upstream request quota.
    for video_id in selected:
        candidates = _media_candidates(media_dir, video_id)
        valid = [path for path in candidates if _video_is_decodable(path)]
        if len(valid) == 1:
            outcomes[video_id] = valid[0]
            failures.pop(video_id, None)
        else:
            pending.append(video_id)

    manifest_order = {video_id: index for index, video_id in enumerate(selected)}
    pending.sort(
        key=lambda video_id: (
            _download_failure_priority(failures.get(video_id)),
            manifest_order[video_id],
        )
    )
    completed = len(outcomes)
    rate_limited = False

    def write_failure_log() -> None:
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(
            "".join(
                json.dumps({"video_id": video_id, "error": failures[video_id]}) + "\n"
                for video_id in sorted(failures)
            ),
            encoding="utf-8",
        )

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    future_to_id = {
        pool.submit(
            _download_video,
            video_id,
            media_dir,
            yt_dlp=yt_dlp,
            cookies_from_browser=cookies_from_browser,
            retries=retries,
            concurrent_fragments=concurrent_fragments,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            extra_args=extra_args,
            fallback_source=source_overrides.get(video_id),
        ): video_id
        for video_id in pending
    }
    try:
        for future in as_completed(future_to_id):
            video_id = future_to_id[future]
            try:
                outcomes[video_id] = future.result()
                failures.pop(video_id, None)
            except DownloadFailure as error:
                failures[video_id] = str(error)
            completed += 1
            if completed % 25 == 0 or completed == len(selected):
                write_failure_log()
                print(
                    f"download progress={completed}/{len(selected)} "
                    f"ok={len(outcomes)} failed={len(failures)}"
                )
            if video_id in failures and _is_youtube_bot_gate(failures[video_id]):
                rate_limited = True
                print(
                    "YouTube bot-confirmation gate detected; stopping this pass "
                    "without classifying the remaining IDs as failures."
                )
                break
    finally:
        interrupted = completed != len(selected)
        if interrupted:
            for future in future_to_id:
                future.cancel()
        write_failure_log()
        pool.shutdown(wait=rate_limited or not interrupted, cancel_futures=interrupted)

    print(f"Wrote {failure_log}")
    return outcomes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_source_url(
    video_id: str, source_overrides: dict[str, dict[str, str]]
) -> str:
    source = source_overrides.get(video_id)
    return (
        source["url"] if source is not None else _YOUTUBE_URL.format(video_id=video_id)
    )


def write_media_manifest(
    video_ids: tuple[str, ...],
    media: dict[str, Path],
    destination: Path,
    source_overrides: dict[str, dict[str, str]] | None = None,
) -> None:
    source_overrides = source_overrides or load_media_source_overrides()
    missing = sorted(set(video_ids) - set(media))
    if missing:
        raise ValueError(
            f"cannot freeze media manifest: {len(missing)} videos are missing; "
            f"first IDs: {missing[:10]}"
        )
    destination.write_text(
        "".join(
            json.dumps(
                {
                    "video_id": video_id,
                    "source_url": _YOUTUBE_URL.format(video_id=video_id),
                    "media_source_url": _media_source_url(video_id, source_overrides),
                    "filename": media[video_id].name,
                    "bytes": media[video_id].stat().st_size,
                    "sha256": _sha256_file(media[video_id]),
                },
                sort_keys=True,
            )
            + "\n"
            for video_id in video_ids
        ),
        encoding="utf-8",
    )


def _video_id_from_path(path: Path) -> str | None:
    if _VIDEO_ID.fullmatch(path.stem):
        return path.stem
    if _VIDEO_ID.fullmatch(path.parent.name):
        return path.parent.name
    return None


def index_media(media_root: Path) -> dict[str, Path]:
    if not media_root.is_dir():
        raise ValueError(f"media root is not a directory: {media_root}")
    media: dict[str, Path] = {}
    for path in sorted(media_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        video_id = _video_id_from_path(path)
        if video_id is None:
            continue
        if video_id in media:
            raise ValueError(
                f"multiple media files resolve to {video_id}: "
                f"{media[video_id]} and {path}"
            )
        media[video_id] = path.resolve()
    return media


def build_datasets(
    protocol: Protocol,
    media: dict[str, Path],
    source_overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    source_overrides = source_overrides or load_media_source_overrides()
    required = set(protocol.queries) | set(protocol.database)
    missing = sorted(required - set(media))
    if missing:
        raise ValueError(
            f"missing {len(missing)} protocol videos under the media root; "
            f"first IDs: {missing[:10]}"
        )

    from datasets import Dataset, Features, Sequence, Value, Video

    corpus = Dataset.from_dict(
        {
            "id": list(protocol.database),
            "video": [str(media[video_id]) for video_id in protocol.database],
            "source_url": [
                _YOUTUBE_URL.format(video_id=video_id) for video_id in protocol.database
            ],
            "media_source_url": [
                _media_source_url(video_id, source_overrides)
                for video_id in protocol.database
            ],
        },
        features=Features(
            {
                "id": Value("string"),
                "video": Video(),
                "source_url": Value("string"),
                "media_source_url": Value("string"),
            }
        ),
    )
    queries = Dataset.from_dict(
        {
            "id": list(protocol.queries),
            "video": [str(media[video_id]) for video_id in protocol.queries],
            "source_url": [
                _YOUTUBE_URL.format(video_id=video_id) for video_id in protocol.queries
            ],
            "media_source_url": [
                _media_source_url(video_id, source_overrides)
                for video_id in protocol.queries
            ],
            "event": [protocol.query_events[video_id] for video_id in protocol.queries],
            "ignored_corpus_ids": [
                list(protocol.query_ignored[video_id]) for video_id in protocol.queries
            ],
        },
        features=Features(
            {
                "id": Value("string"),
                "video": Video(),
                "source_url": Value("string"),
                "media_source_url": Value("string"),
                "event": Value("string"),
                "ignored_corpus_ids": Sequence(Value("string")),
            }
        ),
    )
    qrels = Dataset.from_dict(
        {
            "query-id": [query_id for query_id, _, _ in protocol.qrels],
            "corpus-id": [corpus_id for _, corpus_id, _ in protocol.qrels],
            "score": [score for _, _, score in protocol.qrels],
        },
        features=Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("int32"),
            }
        ),
    )
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def export_local_dataset(
    protocol: Protocol, media: dict[str, Path], output_dir: Path
) -> None:
    from datasets import DatasetDict

    datasets = build_datasets(protocol, media)
    output_dir.mkdir(parents=True, exist_ok=False)
    for config, dataset in datasets.items():
        DatasetDict({"test": dataset}).save_to_disk(output_dir / config)


def _event_coverage_table(summary: dict[str, Any]) -> str:
    rows = [
        "| Event | Original Q/P | S2VS Q/P | Packaged Q/P | Removed from S2VS Q/P |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(
        "| {event} | {original_queries}/{original_positives} | "
        "{before_filter_queries}/{before_filter_positives} | "
        "{queries}/{positives} | {removed_queries}/{removed_positives} |".format(
            **event
        )
        for event in summary["events"]
    )
    return "\n".join(rows)


def dataset_card(summary: dict[str, Any], media_manifest_sha256: str) -> str:
    evaluation = summary["evaluation_protocol"]
    before_filter = summary["before_filter_s2vs_protocol"]
    coverage_table = _event_coverage_table(summary)
    return f"""---
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
license: unknown
pretty_name: EVVE Video-to-Video Event Retrieval
---

# EVVE Video-to-Video Event Retrieval

This repository packages a fixed, self-contained evaluation set derived from
the EVent VidEo (EVVE) benchmark introduced by Revaud et al. at CVPR 2013.
Given a query video of a specific real-world event, the task is to retrieve
other videos depicting the same event.

## Original and packaged protocols

The published EVVE core contains 620 queries, 2,375 database videos, 13 events,
135,213 positive query/database judgments, and 166 hours of web video. The
original large-scale experiment additionally used 100,000 distractor videos.

The public S2VS EVVE feature artifact identifies a surviving 2,410-video layer
of the original core. A reproducible acquisition pass obtained 2,110 of those
videos from public media sources: 2,042 directly from their original YouTube
IDs and 68 from checksum-pinned public archive captures (67 Wayback Machine
captures and one Internet Archive item). Intersecting the 2,110-video manifest
with the checksum-pinned annotation snapshot yields:

- {evaluation["queries"]:,} query videos (154 fewer than the original core);
- {evaluation["database"]:,} database videos (731 fewer than the original core);
- {evaluation["events"]} events;
- {evaluation["qrels"]:,} positive query/database relevance judgments;
- {evaluation["positive_database_videos"]:,} database videos relevant to at
  least one event and {evaluation["other_database_videos"]:,} other core
  candidates;
- every retained query has at least one positive;
- zero query/corpus ID overlap.

The acquisition filter removed 300 IDs from the S2VS layer: 38 queries and 262
database videos (81 positive and 181 other database videos), eliminating 13,864
qrels. At freeze time, yt-dlp reported 95 private videos, 64 videos from
terminated accounts, 14 age-restricted videos, four copyright-claimed or
blocked videos, one terms-of-service removal, and 122 otherwise unavailable
videos. Exact attrition is reproducible as the set difference between
`construction/s2vs-2023-video-ids.txt` ({before_filter["queries"] + before_filter["database"]:,}
IDs) and `construction/surviving-public-media-video-ids.txt`
({evaluation["queries"] + evaluation["database"]:,} IDs).

`media-manifest.jsonl` records the exact frozen ID, packaged filename, original
and acquisition URLs, byte count, and SHA-256 checksum for every video. The
manifest's own SHA-256 is `{media_manifest_sha256}`. Construction fails if any
frozen video is missing or contains no decodable frames, so later source
availability cannot silently alter the task. The complete frozen tree also
passed MTEB's TorchCodec 10-frame uniform sampler. `media-audit.json` records
final completeness and affected-qrel counts.

## Event coverage

Q/P means retained queries / positive database videos. All 13 official events
remain represented, and every packaged query has one or more packaged positives.

{coverage_table}

## Construction and relevance

The source metadata identifies videos by YouTube ID. The construction script
downloads video at up to 360p with yt-dlp, without transcoding. Four IDs whose
progressive MP4 currently contains no media payload use separate complete
video-only and audio streams, which yt-dlp merges without transcoding. For
source IDs no longer downloadable from YouTube, checksum-pinned
public archive captures in `construction/media-source-overrides.json` are used.
The original and actual acquisition URLs remain separate in every row and in
`media-manifest.jsonl`. The resulting media is embedded in the Hugging Face
dataset. The `corpus`, `queries`, and `qrels` configurations use the standard
MTEB retrieval schema. Query rows also retain the EVVE event name and
event-specific ignored (`null`) database IDs required by the original evaluator
(the pinned annotation snapshot contains no retained `null` IDs).

For every retained query, positive qrels are the Cartesian product with all
retained positive database videos for the same official EVVE event. The
100,000-video distractor collection is not included because the original
distractor media is no longer publicly available.

This is a reproducible surviving-public-media subset of EVVE, not the complete
original benchmark. Because both the core corpus and distractor protocol differ,
published/source-paper scores are not directly comparable with results here.

## Media provenance and license

The media originated from the YouTube IDs published by EVVE; public archival
captures are used only where listed explicitly in the source-override manifest.
No applicable dataset-wide redistribution license for the original videos was
identified; the archived project page also contains no explicit third-party
redistribution prohibition. Copyright remains with the respective video rights
holders. The license is therefore recorded as **not specified**. This provenance
statement does not claim that video copyright was transferred to EVVE or to
this repository, or that redistribution is authorized.

The EVVE project page states separately that its software is BSD-licensed and
its released descriptors are free. Those terms are not presented here as a
license for the packaged videos.

## Sources

- [Archived EVVE project page](https://web.archive.org/web/20241102150758id_/http://pascal.inrialpes.fr/data2/evve/index.html)
- [Original paper](https://openaccess.thecvf.com/content_cvpr_2013/html/Revaud_Event_Retrieval_in_2013_CVPR_paper.html)
- [Pinned S2VS annotations](https://github.com/gkordo/s2vs/tree/{ANNOTATIONS_REVISION})
- [S2VS EVVE feature artifact]({FEATURES_URL})

## Citation

```bibtex
@inproceedings{{revaud2013event,
  author = {{Revaud, Jerome and Douze, Matthijs and Schmid, Cordelia and Jegou, Herve}},
  booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
  pages = {{2459--2466}},
  title = {{Event Retrieval in Large Video Collections with Circulant Temporal Encoding}},
  year = {{2013}}
}}
```
"""


def push_dataset(
    protocol: Protocol,
    media: dict[str, Path],
    *,
    repo_id: str,
    work_dir: Path,
    summary: dict[str, Any],
    media_manifest_path: Path,
) -> str:
    from datasets import DatasetDict
    from huggingface_hub import HfApi

    api = HfApi(token=True)
    identity = api.whoami()
    username = identity["name"]
    namespace = repo_id.split("/", 1)[0]
    organizations = {organization["name"] for organization in identity.get("orgs", [])}
    if namespace != username and namespace not in organizations:
        raise ValueError(
            f"authenticated Hugging Face identity {username!r} cannot publish to "
            f"namespace {namespace!r}"
        )

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    for config, dataset in build_datasets(protocol, media).items():
        DatasetDict({"test": dataset}).push_to_hub(
            repo_id,
            config,
            max_shard_size="1GB",
            embed_external_files=True,
            commit_message=f"Add EVVE {config}",
        )

    media_manifest_sha256 = _sha256_file(media_manifest_path)
    card_path = work_dir / "README.md"
    card_path.write_text(dataset_card(summary, media_manifest_sha256), encoding="utf-8")
    summary_path = work_dir / "protocol-summary.json"
    media_audit_path = work_dir / "media-audit.json"
    for local_path, path_in_repo in (
        (card_path, "README.md"),
        (Path(__file__), "construction/create_data.py"),
        (
            PROTOCOL_MANIFEST,
            f"construction/{PROTOCOL_MANIFEST.name}",
        ),
        (
            S2VS_PROTOCOL_MANIFEST,
            f"construction/{S2VS_PROTOCOL_MANIFEST.name}",
        ),
        (
            MEDIA_SOURCE_OVERRIDES,
            f"construction/{MEDIA_SOURCE_OVERRIDES.name}",
        ),
        (media_manifest_path, media_manifest_path.name),
        (media_audit_path, media_audit_path.name),
        (summary_path, summary_path.name),
    ):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path_in_repo}",
        )

    revision = api.dataset_info(repo_id).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    api.update_repo_settings(repo_id, repo_type="dataset", private=False)
    (work_dir / "hub_revision.txt").write_text(revision + "\n", encoding="utf-8")
    return str(revision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/evve-mteb"))
    parser.add_argument("--repo-id", default="Cerru02/EVVE")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Use an existing annotation pickle (the pinned checksum is still required)",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Existing media root (default for downloads: WORK_DIR/media)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local MTEB dataset output (default: WORK_DIR/mteb_export)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download every frozen YouTube ID with yt-dlp",
    )
    parser.add_argument(
        "--download-id-file",
        type=Path,
        help=(
            "Download only these frozen IDs for a distributed/resume pass; final "
            "packaging still validates the complete protocol"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrent-fragments", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-interval", type=float, default=0.0)
    parser.add_argument("--max-sleep-interval", type=float, default=1.0)
    parser.add_argument(
        "--yt-dlp-arg",
        action="append",
        default=[],
        help=(
            "Additional argument passed verbatim to each yt-dlp invocation; repeat "
            "the option for arguments that take a value"
        ),
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Pass through to yt-dlp, for example chrome, firefox, or safari",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Download only the first N IDs for a smoke test; never package a subset",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write the failure log without failing the download-only invocation",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = args.annotations or fetch_annotations(args.work_dir)
    annotations = load_annotations(annotation_path)
    before_filter = build_protocol(
        annotations,
        load_protocol_ids(S2VS_PROTOCOL_MANIFEST),
        enforce_expected_counts=False,
    )
    before_counts = _protocol_counts(before_filter)
    expected_before_counts = (S2VS_QUERIES, S2VS_DATABASE, S2VS_QRELS)
    actual_before_counts = (
        before_counts["queries"],
        before_counts["database"],
        before_counts["qrels"],
    )
    if actual_before_counts != expected_before_counts:
        raise ValueError(
            "S2VS protocol counts changed: "
            f"{actual_before_counts}; expected {expected_before_counts}"
        )
    protocol = build_protocol(annotations, load_protocol_ids())
    summary = protocol_summary(protocol, before_filter)
    summary_path = args.work_dir / "protocol-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["evaluation_protocol"], indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")

    media_root = (args.media_root or args.work_dir / "media").resolve()
    if args.download:
        protocol_ids = load_protocol_ids()
        download_ids = (
            load_download_ids(args.download_id_file, protocol_ids)
            if args.download_id_file is not None
            else protocol_ids
        )
        download_media(
            download_ids,
            media_root,
            workers=args.workers,
            cookies_from_browser=args.cookies_from_browser,
            retries=args.retries,
            concurrent_fragments=args.concurrent_fragments,
            sleep_min=args.sleep_interval,
            sleep_max=args.max_sleep_interval,
            extra_args=tuple(args.yt_dlp_arg),
            source_overrides=load_media_source_overrides(),
            limit=args.limit,
            failure_log=args.work_dir / "download-failures.jsonl",
        )

    should_package = (
        args.push or args.output_dir is not None or args.media_root is not None
    )
    if not args.download and not should_package:
        print("Metadata audit complete; no media was read or downloaded.")
        return

    media = index_media(media_root)
    required = set(protocol.queries) | set(protocol.database)
    missing = sorted(required - set(media))
    invalid = sorted(
        video_id
        for video_id in required & set(media)
        if not _video_is_decodable(media[video_id])
    )
    audit = media_audit(protocol, media, set(invalid))
    media_audit_path = args.work_dir / "media-audit.json"
    media_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "events"}))
    print(f"Wrote {media_audit_path}")
    if missing or invalid:
        if args.allow_incomplete and not should_package:
            return
        raise SystemExit(
            f"Frozen protocol is incomplete: missing={len(missing)} invalid={len(invalid)}. "
            f"See {args.work_dir / 'download-failures.jsonl'} and rerun to resume."
        )
    if args.limit is not None:
        raise SystemExit(
            "--limit is for download smoke tests and cannot package a dataset"
        )

    media_manifest_path = args.work_dir / "media-manifest.jsonl"
    write_media_manifest(
        load_protocol_ids(),
        media,
        media_manifest_path,
        load_media_source_overrides(),
    )
    print(f"Wrote {media_manifest_path}")

    if args.push:
        revision = push_dataset(
            protocol,
            media,
            repo_id=args.repo_id,
            work_dir=args.work_dir,
            summary=summary,
            media_manifest_path=media_manifest_path,
        )
        print(f"Pushed {args.repo_id} @ {revision}")
    elif should_package:
        output_dir = args.output_dir or args.work_dir / "mteb_export"
        export_local_dataset(protocol, media, output_dir)
        print(f"Wrote local MTEB datasets to {output_dir}")


if __name__ == "__main__":
    main()
