#!/usr/bin/env python3
"""Build the VCDB core set as MTEB video and audio retrieval datasets.

The source is the official multipart ``core_dataset`` archive distributed by
Fudan University. Download the twelve parts manually from the VCDB page so the
research-only terms are presented by the source site, then run:

    uv run python scripts/data/vcdb_retrieval/create_data.py \
        --source-dir /path/to/archive-parts \
        --work-dir /tmp/vcdb-core

The command audits and exports the dataset locally. Publishing is deliberately
opt-in and requires a logged-in Hugging Face account:

    uv run python scripts/data/vcdb_retrieval/create_data.py \
        --source-dir /path/to/archive-parts \
        --work-dir /tmp/vcdb-core \
        --repo-id pranitchawla/VCDB-Core \
        --push

Audio extraction is published separately so the original video task remains
unchanged and the one source video without an audio stream can be excluded:

    uv run python scripts/data/vcdb_retrieval/create_data.py \
        --source-dir /path/to/archive-parts \
        --work-dir /tmp/vcdb-core \
        --audio-repo-id pranitchawla/VCDB-Core-Audio \
        --push-audio
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Value, Video
from huggingface_hub import HfApi, create_repo, get_token

_SOURCE_URL = "https://fvl.fudan.edu.cn/dataset/vcdb/list.htm"
_ARCHIVED_TERMS_URL = (
    "https://web.archive.org/web/20251123150707/"
    "https://fvl.fudan.edu.cn/dataset/vcdb/list.htm"
)
_PAPER_URL = (
    "https://fvl.fudan.edu.cn/_upload/article/files/7b/c4/"
    "190424104d2192e8e83cb9dfa6fc/5a8d85f7-5738-450c-b957-3de71d8d7e72.pdf"
)
_REFERENCE_SETUP_URL = (
    "https://github.com/facebookresearch/videoalignment/blob/main/VCDB.sh"
)

_EXPECTED_VIDEO_COUNT = 528
_EXPECTED_AUDIO_COUNT = 527
_EXPECTED_ANNOTATION_GROUPS = 28
_EXPECTED_RAW_ANNOTATIONS = 9236
_ARCHIVE_PARTS = [*(f"core_dataset.z{i:02d}" for i in range(1, 12)), "core_dataset.zip"]
_VIDEO_SUFFIXES = {".flv", ".mp4"}

# The public facebookresearch/videoalignment VCDB setup removes these records.
# They are retained in the raw-count audit and excluded from the processed data.
_KNOWN_INVALID_ANNOTATIONS = {
    (
        "38f11a4162d8e94227ac644f117f942735b9a504.mp4",
        "bf582249cfc79d691195a8681961029cc5149a76.flv",
        "00:02:35",
        "00:03:15",
        "00:01:04",
        "00:03:18",
    ),
    (
        "14c262ea09b4ca66feb7e88cf57e0faaeacc301f.mp4",
        "f150e062960b477adcac3f12ef4543337f5a91a4.flv",
        "00:00:19",
        "00:00:33",
        "00:00:00",
        "00:00:14",
    ),
    (
        "067fb2aa9623905a42a2b0b286de1386e45c5bf8.flv",
        "8f19329946455ae5c2c7b788ea6f6513bf5e1c9a.flv",
        "00:01:06",
        "00:02:18",
        "00:00:05",
        "00:01:17",
    ),
}

_BIBTEX = r"""
@inproceedings{jiang2014vcdb,
  author = {Jiang, Yu-Gang and Jiang, Yudong and Wang, Jiajun},
  booktitle = {Computer Vision -- ECCV 2014},
  pages = {357--371},
  publisher = {Springer},
  title = {VCDB: A Large-Scale Database for Partial Copy Detection in Videos},
  year = {2014},
}
""".strip()


@dataclass(frozen=True)
class Annotation:
    """One official segment-level copy annotation."""

    annotation_group: str
    source_row: int
    video_a_id: str
    video_b_id: str
    start_a_seconds: int
    end_a_seconds: int
    start_b_seconds: int
    end_b_seconds: int


@dataclass(frozen=True)
class VideoInfo:
    """Validated video metadata and the local path used for packaging."""

    id: str
    path: str
    duration_seconds: float
    width: int
    height: int
    codec: str
    transcoded: bool


@dataclass(frozen=True)
class AudioInfo:
    """Validated extracted audio metadata and its source video ID."""

    id: str
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    source_codec: str
    packaged_codec: str
    reencoded: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_parts(source_dir: Path) -> dict[str, dict[str, Any]]:
    missing = [name for name in _ARCHIVE_PARTS if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing VCDB archive parts: " + ", ".join(missing))
    return {
        name: {
            "size_bytes": (source_dir / name).stat().st_size,
            "sha256": _sha256(source_dir / name),
        }
        for name in _ARCHIVE_PARTS
    }


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            output = (destination / member.filename).resolve()
            if (
                destination_resolved not in output.parents
                and output != destination_resolved
            ):
                raise RuntimeError(f"Unsafe path in VCDB archive: {member.filename}")
        bundle.extractall(destination)


def _extract_source(source_dir: Path, work_dir: Path) -> tuple[Path, Path]:
    extracted = work_dir / "source"
    videos_dir = extracted / "core_dataset"
    annotations_dir = extracted / "annotation"
    if videos_dir.is_dir() and annotations_dir.is_dir():
        return videos_dir, annotations_dir

    joined_archive = work_dir / "core_dataset.joined.zip"
    if not joined_archive.is_file():
        zip_bin = shutil.which("zip")
        if zip_bin is None:
            raise RuntimeError(
                "The `zip` executable is required for multipart archives"
            )
        subprocess.run(
            [
                zip_bin,
                "-s",
                "0",
                str(source_dir / "core_dataset.zip"),
                "--out",
                str(joined_archive),
            ],
            check=True,
        )
    _safe_extract_zip(joined_archive, extracted)
    if not videos_dir.is_dir() or not annotations_dir.is_dir():
        raise RuntimeError(
            "Unexpected VCDB archive layout: expected `core_dataset/` and `annotation/`"
        )
    return videos_dir, annotations_dir


def _timestamp_to_seconds(timestamp: str) -> int:
    parts = timestamp.strip().split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid VCDB timestamp: {timestamp!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid VCDB timestamp: {timestamp!r}")
    return hours * 3600 + minutes * 60 + seconds


def _load_annotations(
    annotations_dir: Path, videos: list[VideoInfo]
) -> tuple[list[Annotation], list[dict[str, Any]], list[dict[str, Any]], int]:
    files = sorted(annotations_dir.glob("*.txt"))
    if len(files) != _EXPECTED_ANNOTATION_GROUPS:
        raise RuntimeError(
            f"Expected {_EXPECTED_ANNOTATION_GROUPS} annotation groups, got {len(files)}"
        )

    annotations: list[Annotation] = []
    excluded: list[dict[str, Any]] = []
    media_name_normalizations: list[dict[str, Any]] = []
    media_by_group_and_stem: dict[tuple[str, str], str] = {}
    for video in videos:
        relative_path = Path(video.id)
        key = (relative_path.parent.as_posix(), relative_path.stem)
        if key in media_by_group_and_stem:
            raise RuntimeError(f"Ambiguous group-local media stem: {key}")
        media_by_group_and_stem[key] = video.id

    raw_count = 0
    for path in files:
        group = path.stem
        with path.open(encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                raw_count += 1
                normalized = tuple(value.strip() for value in row)
                if len(normalized) != 6:
                    raise RuntimeError(
                        f"Expected 6 fields in {path.name}:{row_number}, got {len(row)}"
                    )
                if normalized in _KNOWN_INVALID_ANNOTATIONS:
                    excluded.append(
                        {
                            "annotation_group": group,
                            "source_row": row_number,
                            "raw": list(normalized),
                            "reason": "Excluded by the public facebookresearch VCDB setup",
                        }
                    )
                    continue

                media_keys = [(group, Path(normalized[index]).stem) for index in (0, 1)]
                missing = [
                    key for key in media_keys if key not in media_by_group_and_stem
                ]
                if missing:
                    raise RuntimeError(
                        f"Annotation references missing media in {path.name}:{row_number}: {missing}"
                    )
                video_a_id, video_b_id = (
                    media_by_group_and_stem[key] for key in media_keys
                )
                for raw_name, resolved_id in zip(
                    normalized[:2], (video_a_id, video_b_id), strict=True
                ):
                    if Path(resolved_id).name != raw_name:
                        media_name_normalizations.append(
                            {
                                "annotation_group": group,
                                "source_row": row_number,
                                "annotation_filename": raw_name,
                                "resolved_video_id": resolved_id,
                            }
                        )
                start_a = _timestamp_to_seconds(normalized[2])
                end_a = _timestamp_to_seconds(normalized[3])
                start_b = _timestamp_to_seconds(normalized[4])
                end_b = _timestamp_to_seconds(normalized[5])
                if end_a <= start_a or end_b <= start_b:
                    raise RuntimeError(
                        f"Non-positive segment duration in {path.name}:{row_number}"
                    )
                annotations.append(
                    Annotation(
                        annotation_group=group,
                        source_row=row_number,
                        video_a_id=video_a_id,
                        video_b_id=video_b_id,
                        start_a_seconds=start_a,
                        end_a_seconds=end_a,
                        start_b_seconds=start_b,
                        end_b_seconds=end_b,
                    )
                )

    if raw_count != _EXPECTED_RAW_ANNOTATIONS:
        raise RuntimeError(
            f"Expected {_EXPECTED_RAW_ANNOTATIONS} raw annotations, got {raw_count}"
        )
    missing_known_invalid = len(_KNOWN_INVALID_ANNOTATIONS) - len(excluded)
    if missing_known_invalid:
        raise RuntimeError(
            f"Expected all 3 known invalid annotations, missing {missing_known_invalid}"
        )
    return annotations, excluded, media_name_normalizations, raw_count


def _probe_video(path: Path) -> tuple[float, int, int, str] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate VCDB media")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
        duration = float(stream.get("duration") or payload["format"]["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        codec = str(stream["codec_name"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if duration <= 0 or width <= 0 or height <= 0 or not codec:
        return None
    return duration, width, height, codec


def _is_decodable(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to validate VCDB media")
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0


def _transcode_video(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to transcode incompatible VCDB media")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        check=True,
    )


def _validate_one_video(
    video_id: str, source_path: Path, transcode_dir: Path
) -> VideoInfo:
    probe = _probe_video(source_path)
    transcoded = False
    packaged_path = source_path
    if probe is None or not _is_decodable(source_path):
        packaged_path = transcode_dir / Path(video_id).with_suffix(".mp4")
        _transcode_video(source_path, packaged_path)
        probe = _probe_video(packaged_path)
        transcoded = True
        if probe is None or not _is_decodable(packaged_path):
            raise RuntimeError(f"Video is not decodable after transcoding: {video_id}")
    if probe is None:
        raise RuntimeError(f"Video is not decodable after transcoding: {video_id}")
    duration, width, height, codec = probe
    return VideoInfo(
        id=video_id,
        path=str(packaged_path),
        duration_seconds=duration,
        width=width,
        height=height,
        codec=codec,
        transcoded=transcoded,
    )


def _load_media(videos_dir: Path, work_dir: Path, workers: int) -> list[VideoInfo]:
    media_paths = sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
    )
    if len(media_paths) != _EXPECTED_VIDEO_COUNT:
        raise RuntimeError(
            f"Expected {_EXPECTED_VIDEO_COUNT} core videos, got {len(media_paths)}"
        )
    media = [(path.relative_to(videos_dir).as_posix(), path) for path in media_paths]
    ids = [video_id for video_id, _ in media]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate stable video IDs in VCDB core")

    transcode_dir = work_dir / "transcoded"
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        infos = list(
            pool.map(
                lambda item: _validate_one_video(item[0], item[1], transcode_dir),
                media,
            )
        )
    return sorted(infos, key=lambda info: info.id)


def _probe_audio(path: Path) -> tuple[float, int, int, str] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate VCDB audio")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
        duration = float(stream.get("duration") or payload["format"]["duration"])
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec = str(stream["codec_name"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if duration <= 0 or sample_rate <= 0 or channels <= 0 or not codec:
        return None
    return duration, sample_rate, channels, codec


def _is_audio_decodable(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to validate VCDB audio")
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0


def _extract_one_audio(
    video_id: str, source_path: Path, audio_dir: Path
) -> AudioInfo | None:
    source_probe = _probe_audio(source_path)
    if source_probe is None:
        return None
    _, _, _, source_codec = source_probe

    destination = audio_dir / Path(video_id).with_suffix(".m4a")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reencoded = source_codec != "aac"
    if destination.is_file():
        packaged_probe = _probe_audio(destination)
        if packaged_probe is not None and _is_audio_decodable(destination):
            duration, sample_rate, channels, packaged_codec = packaged_probe
            return AudioInfo(
                id=video_id,
                path=str(destination),
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                source_codec=source_codec,
                packaged_codec=packaged_codec,
                reencoded=reencoded,
            )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract VCDB audio")
    partial = destination.with_name(f".{destination.stem}.partial.m4a")
    partial.unlink(missing_ok=True)
    codec_args = (
        ["-c:a", "aac", "-b:a", "128k", "-threads", "1"]
        if reencoded
        else ["-c:a", "copy"]
    )
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            *codec_args,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            "-y",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not _is_audio_decodable(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to extract a decodable audio track: {video_id}")
    partial.replace(destination)
    packaged_probe = _probe_audio(destination)
    if packaged_probe is None:
        raise RuntimeError(f"Failed to probe extracted audio track: {video_id}")
    duration, sample_rate, channels, packaged_codec = packaged_probe
    return AudioInfo(
        id=video_id,
        path=str(destination),
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        source_codec=source_codec,
        packaged_codec=packaged_codec,
        reencoded=reencoded,
    )


def _extract_audio_tracks(
    videos: list[VideoInfo], videos_dir: Path, work_dir: Path, workers: int
) -> tuple[list[AudioInfo], list[str]]:
    audio_dir = work_dir / "audio"
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(
            pool.map(
                lambda video: _extract_one_audio(
                    video.id, videos_dir / video.id, audio_dir
                ),
                videos,
            )
        )
    audio = sorted(
        (item for item in results if item is not None), key=lambda item: item.id
    )
    missing = sorted(
        video.id for video, item in zip(videos, results, strict=True) if item is None
    )
    if len(audio) != _EXPECTED_AUDIO_COUNT:
        raise RuntimeError(
            f"Expected {_EXPECTED_AUDIO_COUNT} audio tracks, got {len(audio)}; "
            f"missing={missing}"
        )
    if len({item.id for item in audio}) != len(audio):
        raise RuntimeError("Duplicate stable IDs in extracted VCDB audio")
    return audio, missing


def _audit_annotation_durations(
    annotations: list[Annotation], videos: list[VideoInfo]
) -> list[dict[str, Any]]:
    durations = {video.id: video.duration_seconds for video in videos}
    findings: list[dict[str, Any]] = []
    for annotation in annotations:
        # VCDB timestamps are integer seconds. Permit one second of container
        # rounding at the end boundary.
        if annotation.end_a_seconds > durations[annotation.video_a_id] + 1:
            findings.append(
                {
                    "annotation_group": annotation.annotation_group,
                    "source_row": annotation.source_row,
                    "side": "video_a",
                    "video_id": annotation.video_a_id,
                    "end_seconds": annotation.end_a_seconds,
                    "video_duration_seconds": durations[annotation.video_a_id],
                }
            )
        if annotation.end_b_seconds > durations[annotation.video_b_id] + 1:
            findings.append(
                {
                    "annotation_group": annotation.annotation_group,
                    "source_row": annotation.source_row,
                    "side": "video_b",
                    "video_id": annotation.video_b_id,
                    "end_seconds": annotation.end_b_seconds,
                    "video_duration_seconds": durations[annotation.video_b_id],
                }
            )
    return findings


def _build_datasets(
    videos: list[VideoInfo], annotations: list[Annotation]
) -> tuple[Dataset, Dataset, Dataset, Dataset, dict[str, Any]]:
    video_ids = [video.id for video in videos]
    video_paths = [video.path for video in videos]
    unordered_pairs = sorted(
        {
            tuple(sorted((annotation.video_a_id, annotation.video_b_id)))
            for annotation in annotations
            if annotation.video_a_id != annotation.video_b_id
        }
    )

    directed_pairs = sorted(
        pair
        for video_a, video_b in unordered_pairs
        for pair in ((video_a, video_b), (video_b, video_a))
    )
    query_ids = sorted({query_id for query_id, _ in directed_pairs})
    path_by_id = dict(zip(video_ids, video_paths, strict=True))

    corpus = Dataset.from_dict({"id": video_ids, "video": video_paths}).cast_column(
        "video", Video()
    )
    queries = Dataset.from_dict(
        {"id": query_ids, "video": [path_by_id[id_] for id_ in query_ids]}
    ).cast_column("video", Video())
    qrels = Dataset.from_dict(
        {
            "query-id": [query_id for query_id, _ in directed_pairs],
            "corpus-id": [corpus_id for _, corpus_id in directed_pairs],
            "score": [1] * len(directed_pairs),
        }
    ).cast_column("score", Value("int32"))
    annotation_dataset = Dataset.from_list([asdict(item) for item in annotations])

    corpus_ids = set(video_ids)
    qrel_set = set(directed_pairs)
    if len(qrel_set) != len(directed_pairs):
        raise RuntimeError("Duplicate directed qrels after collapsing annotations")
    if any(
        query_id not in corpus_ids or doc_id not in corpus_ids
        for query_id, doc_id in qrel_set
    ):
        raise RuntimeError("Qrel references an ID outside the corpus")
    if any(query_id == doc_id for query_id, doc_id in qrel_set):
        raise RuntimeError("Self relation found in qrels")
    if any((doc_id, query_id) not in qrel_set for query_id, doc_id in qrel_set):
        raise RuntimeError("Qrels are not symmetric")
    qrels_per_query = Counter(query_id for query_id, _ in directed_pairs)
    if set(qrels_per_query) != set(query_ids) or any(
        count < 1 for count in qrels_per_query.values()
    ):
        raise RuntimeError("Every query must have at least one positive")

    summary = {
        "corpus_videos": len(video_ids),
        "queries": len(query_ids),
        "segment_annotations": len(annotations),
        "self_segment_annotations_excluded_from_qrels": sum(
            annotation.video_a_id == annotation.video_b_id for annotation in annotations
        ),
        "unique_undirected_video_pairs": len(unordered_pairs),
        "directed_qrels": len(directed_pairs),
        "implicit_negatives_per_query_min": min(
            len(video_ids) - 1 - count for count in qrels_per_query.values()
        ),
        "implicit_negatives_per_query_max": max(
            len(video_ids) - 1 - count for count in qrels_per_query.values()
        ),
        "positives_per_query": dict(sorted(Counter(qrels_per_query.values()).items())),
    }
    return corpus, queries, qrels, annotation_dataset, summary


def _build_audio_datasets(
    audio: list[AudioInfo], annotations: list[Annotation]
) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    audio_ids = [item.id for item in audio]
    audio_id_set = set(audio_ids)
    path_by_id = {item.id: item.path for item in audio}
    all_unordered_pairs = {
        tuple(sorted((annotation.video_a_id, annotation.video_b_id)))
        for annotation in annotations
        if annotation.video_a_id != annotation.video_b_id
    }
    unordered_pairs = sorted(
        (video_a, video_b)
        for video_a, video_b in all_unordered_pairs
        if video_a in audio_id_set and video_b in audio_id_set
    )
    directed_pairs = sorted(
        pair
        for video_a, video_b in unordered_pairs
        for pair in ((video_a, video_b), (video_b, video_a))
    )
    query_ids = sorted({query_id for query_id, _ in directed_pairs})

    corpus = Dataset.from_dict(
        {"id": audio_ids, "audio": [path_by_id[id_] for id_ in audio_ids]}
    ).cast_column("audio", Audio())
    queries = Dataset.from_dict(
        {"id": query_ids, "audio": [path_by_id[id_] for id_ in query_ids]}
    ).cast_column("audio", Audio())
    qrels = Dataset.from_dict(
        {
            "query-id": [query_id for query_id, _ in directed_pairs],
            "corpus-id": [corpus_id for _, corpus_id in directed_pairs],
            "score": [1] * len(directed_pairs),
        }
    ).cast_column("score", Value("int32"))

    qrel_set = set(directed_pairs)
    if len(qrel_set) != len(directed_pairs):
        raise RuntimeError("Duplicate directed audio qrels")
    if any(
        query_id not in audio_id_set or corpus_id not in audio_id_set
        for query_id, corpus_id in qrel_set
    ):
        raise RuntimeError("Audio qrel references an unavailable track")
    if any(query_id == corpus_id for query_id, corpus_id in qrel_set):
        raise RuntimeError("Self relation found in audio qrels")
    if any((corpus_id, query_id) not in qrel_set for query_id, corpus_id in qrel_set):
        raise RuntimeError("Audio qrels are not symmetric")
    qrels_per_query = Counter(query_id for query_id, _ in directed_pairs)
    if set(qrels_per_query) != set(query_ids) or any(
        count < 1 for count in qrels_per_query.values()
    ):
        raise RuntimeError("Every audio query must have at least one positive")

    summary = {
        "corpus_audio_tracks": len(audio_ids),
        "queries": len(query_ids),
        "unique_undirected_video_pairs": len(unordered_pairs),
        "directed_qrels": len(directed_pairs),
        "directed_qrels_removed_for_missing_audio": 2
        * (len(all_unordered_pairs) - len(unordered_pairs)),
        "implicit_negatives_per_query_min": min(
            len(audio_ids) - 1 - count for count in qrels_per_query.values()
        ),
        "implicit_negatives_per_query_max": max(
            len(audio_ids) - 1 - count for count in qrels_per_query.values()
        ),
        "positives_per_query": dict(sorted(Counter(qrels_per_query.values()).items())),
    }
    return corpus, queries, qrels, summary


def _dataset_card(summary: dict[str, Any]) -> str:
    archive_hashes = "\n".join(
        f"- `{name}`: `{metadata['sha256']}` ({metadata['size_bytes']} bytes)"
        for name, metadata in summary["archive_parts"].items()
    )
    excluded_rows = "\n".join(
        f"- `{item['annotation_group']}.txt:{item['source_row']}`: "
        f"`{','.join(item['raw'])}`"
        for item in summary["excluded_annotations"]
    )
    transcodes = summary["transcoded_videos"]
    transcode_text = (
        "None; all source videos were preserved byte-for-byte."
        if not transcodes
        else "\n".join(
            f"- `{item['video_id']}`: source `{item['source_sha256']}`, "
            f"packaged `{item['packaged_sha256']}`"
            for item in transcodes
        )
    )
    retrieval = summary["retrieval"]
    return f"""---
license: other
pretty_name: VCDB Core Retrieval
task_categories:
- other
tags:
- mteb
- moeb
- video-retrieval
- video-copy-detection
- research-only
---

# VCDB Core Retrieval

This repository packages the **528-video core set** of VCDB as a symmetric
video-to-video retrieval task for MTEB/MOEB. It does not include VCDB's separate
100,000-video background collection. For each query, every other core video not
marked relevant acts as an implicit distractor.

## Terms and provenance

The source dataset is provided by Fudan University for **research purposes
only**. The source authors and Fudan University make no warranties about the
dataset, including non-infringement. Users must review and follow the
[original VCDB terms]({_ARCHIVED_TERMS_URL}). The original page is
[`{_SOURCE_URL}`]({_SOURCE_URL}) and may be intermittently unavailable.

The videos were collected from YouTube and MetaCafe. Rights in the underlying
videos remain with their respective rightsholders. This packaging does not
grant additional rights beyond the source terms.

## Construction

- Raw core videos: {summary["raw_video_count"]}
- Raw annotation groups: {summary["raw_annotation_groups"]}
- Raw segment annotations: {summary["raw_annotation_count"]}
- Retained segment annotations: {retrieval["segment_annotations"]}
- Query videos: {retrieval["queries"]}
- Corpus videos: {retrieval["corpus_videos"]}
- Unique undirected relevant video pairs: {retrieval["unique_undirected_video_pairs"]}
- Directed binary qrels: {retrieval["directed_qrels"]}

Multiple copied segments between the same parent-video pair are collapsed into
one binary judgment. Every pair is emitted in both directions. Query and corpus
IDs match for the same physical video; the MTEB task removes this self-match
before scoring. The {retrieval["self_segment_annotations_excluded_from_qrels"]}
diagonal segment annotations are retained in `annotations` but do not create
self-relevance judgments.

One source video is stored as `.mp4` although six annotation references name
the same group-local hash with a `.flv` suffix. These references are resolved
to its actual topic-relative archive path; the audit records every occurrence.

All timestamps have valid `HH:MM:SS` syntax, positive segment lengths, and
non-negative values. The audit records
{len(summary["annotation_duration_findings"])} retained endpoints that exceed
the duration reported by their archived media container. They are preserved
because they are present in the official annotations, are not among the three
rows excluded by the public reference setup, and do not affect the video-level
relationship.

The complete retained segment-level timestamps are preserved in the
`annotations` configuration. Three rows removed by the public
[facebookresearch VCDB setup]({_REFERENCE_SETUP_URL}) are excluded:

{excluded_rows}

### Media conversions

{transcode_text}

Any conversion uses single-threaded FFmpeg/libx264, CRF 18, the medium preset,
`yuv420p`, no audio, no metadata or chapters, and `faststart`.

## Configurations

- `corpus` (`test`): `id: string`, `video: Video`
- `queries` (`test`): `id: string`, `video: Video`
- `qrels` (`test`): `query-id: string`, `corpus-id: string`, `score: int32`
- `annotations` (`test`): source group/row, both video IDs, and both time ranges

## Source archive SHA-256

{archive_hashes}

## Limitations

VCDB was designed for temporal partial-copy localization. This adaptation
evaluates whole-video retrieval and therefore cannot reproduce the paper's
segment-level F1 score. It also uses only the core set, giving a 528-video
corpus rather than the complete 100K-background protocol.

## Citation

```bibtex
{_BIBTEX}
```

Paper: [{_PAPER_URL}]({_PAPER_URL})
"""


def _audio_dataset_card(summary: dict[str, Any]) -> str:
    archive_hashes = "\n".join(
        f"- `{name}`: `{metadata['sha256']}` ({metadata['size_bytes']} bytes)"
        for name, metadata in summary["archive_parts"].items()
    )
    audio_summary = summary["audio"]
    retrieval = audio_summary["retrieval"]
    reencoded_rows = "\n".join(
        f"- `{item['video_id']}`: `{item['source_codec']}` to AAC; "
        f"output SHA-256 `{item['packaged_sha256']}`"
        for item in audio_summary["reencoded_tracks"]
    )
    return f"""---
license: other
pretty_name: VCDB Core Audio Retrieval
task_categories:
- other
tags:
- mteb
- moeb
- audio-retrieval
- duplicate-detection
- research-only
---

# VCDB Core Audio Retrieval

This repository packages audio extracted from the **528-video core set** of
VCDB as a symmetric audio-to-audio retrieval task for MTEB/MOEB. The separate
100,000-video background collection is not included.

## Terms and provenance

The source dataset is provided by Fudan University for **research purposes
only**. The source authors and Fudan University make no warranties about the
dataset, including non-infringement. Users must review and follow the
[original VCDB terms]({_ARCHIVED_TERMS_URL}). The original page is
[`{_SOURCE_URL}`]({_SOURCE_URL}) and may be intermittently unavailable.

The source videos were collected from YouTube and MetaCafe. Rights in the
underlying media remain with their respective rightsholders. This packaging
does not grant additional rights beyond the source terms.

## Construction

- Raw core videos: {summary["raw_video_count"]}
- Videos with an audio stream: {retrieval["corpus_audio_tracks"]}
- Videos without an audio stream: {len(audio_summary["missing_video_ids"])}
- Audio queries: {retrieval["queries"]}
- Unique undirected relevant pairs: {retrieval["unique_undirected_video_pairs"]}
- Directed binary qrels: {retrieval["directed_qrels"]}

Stable IDs remain the original topic-relative video paths. The source video
without audio, `{audio_summary["missing_video_ids"][0]}`, is excluded. Qrels
involving it are removed, and every retained query has at least one positive.

Relevance is derived from VCDB's human segment-level video copy annotations:
multiple annotations for the same source-video pair are collapsed, and both
directions are emitted. These are not independent audio-copy judgments.

## Audio extraction

Source codec counts: `{json.dumps(audio_summary["source_codecs"], sort_keys=True)}`.
Packaged codec counts: `{json.dumps(audio_summary["packaged_codecs"], sort_keys=True)}`.

AAC elementary streams are remuxed without re-encoding into `.m4a`. The
{len(audio_summary["reencoded_tracks"])} non-AAC tracks are deterministically
encoded to AAC at 128 kb/s with one FFmpeg thread, with metadata and chapters
removed. Every packaged track is fully decoded during validation.

Re-encoded tracks:

{reencoded_rows}

## Configurations

- `corpus` (`test`): `id: string`, `audio: Audio`
- `queries` (`test`): `id: string`, `audio: Audio`
- `qrels` (`test`): `query-id: string`, `corpus-id: string`, `score: int32`

## Source archive SHA-256

{archive_hashes}

## Limitations

VCDB was designed for temporal partial-copy detection in video. Copied visual
segments may have modified or replaced sound, so audio-only relevance can be
noisy. This adaptation also uses only the core set and cannot reproduce the
paper's segment-level F1 score or complete 100K-background protocol.

## Citation

```bibtex
{_BIBTEX}
```

Paper: [{_PAPER_URL}]({_PAPER_URL})
"""


def _save_local(
    work_dir: Path,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    annotations: Dataset,
    card: str,
    summary: dict[str, Any],
) -> None:
    export_dir = work_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for name, dataset in {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "annotations": annotations,
    }.items():
        DatasetDict({"test": dataset}).save_to_disk(export_dir / name)
    (export_dir / "README.md").write_text(card, encoding="utf-8")
    (export_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_audio_local(
    work_dir: Path,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    card: str,
    summary: dict[str, Any],
) -> None:
    export_dir = work_dir / "audio_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for name, dataset in {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }.items():
        DatasetDict({"test": dataset}).save_to_disk(export_dir / name)
    (export_dir / "README.md").write_text(card, encoding="utf-8")
    (export_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _publish(
    repo_id: str,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    annotations: Dataset,
    card: str,
    upload_workers: int,
) -> str:
    token = get_token()
    if not token:
        raise RuntimeError(
            "--push requires `hf auth login` or HF_TOKEN in the environment"
        )
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)
    api = HfApi(token=token)
    # Publish the research-only terms before any media. Subsequent dataset pushes
    # enrich this card's YAML with the generated configuration metadata.
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add VCDB Core dataset card and terms",
    )
    for name, dataset in {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "annotations": annotations,
    }.items():
        DatasetDict({"test": dataset}).push_to_hub(
            repo_id,
            name,
            token=token,
            commit_message=f"Add VCDB Core {name} configuration",
            num_proc=max(1, upload_workers),
        )
    return api.dataset_info(repo_id).sha


def _publish_audio(
    repo_id: str,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    card: str,
    summary: dict[str, Any],
    upload_workers: int,
) -> str:
    token = get_token()
    if not token:
        raise RuntimeError(
            "--push-audio requires `hf auth login` or HF_TOKEN in the environment"
        )
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add VCDB Core Audio dataset card and terms",
    )
    for name, dataset in {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }.items():
        DatasetDict({"test": dataset}).push_to_hub(
            repo_id,
            name,
            token=token,
            commit_message=f"Add VCDB Core Audio {name} configuration",
            num_proc=max(1, upload_workers),
        )
    api.upload_file(
        path_or_fileobj=io.BytesIO(
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
        ),
        path_in_repo="audit.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add VCDB Core Audio construction audit",
    )
    return api.dataset_info(repo_id).sha


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing core_dataset.zip and core_dataset.z01..z11",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Scratch/output directory outside the Git repository",
    )
    parser.add_argument("--repo-id", default="pranitchawla/VCDB-Core")
    parser.add_argument("--audio-repo-id", default="pranitchawla/VCDB-Core-Audio")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=1,
        help="Number of parallel workers used only for Hugging Face shard uploads",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Publish the video dataset; requires an authenticated account",
    )
    parser.add_argument(
        "--push-audio",
        action="store_true",
        help="Publish the extracted audio dataset; requires an authenticated account",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    archive_parts = _validate_archive_parts(source_dir)
    videos_dir, annotations_dir = _extract_source(source_dir, work_dir)
    videos = _load_media(videos_dir, work_dir, args.workers)
    audio, missing_audio_ids = _extract_audio_tracks(
        videos, videos_dir, work_dir, args.workers
    )
    annotations, excluded, media_name_normalizations, raw_annotation_count = (
        _load_annotations(annotations_dir, videos)
    )
    annotation_duration_findings = _audit_annotation_durations(annotations, videos)
    corpus, queries, qrels, annotation_dataset, retrieval_summary = _build_datasets(
        videos, annotations
    )
    audio_corpus, audio_queries, audio_qrels, audio_retrieval_summary = (
        _build_audio_datasets(audio, annotations)
    )

    transcoded_videos = [
        {
            "video_id": video.id,
            "source_sha256": _sha256(videos_dir / video.id),
            "packaged_sha256": _sha256(Path(video.path)),
        }
        for video in videos
        if video.transcoded
    ]
    audio_tracks = [
        {
            "video_id": item.id,
            "duration_seconds": item.duration_seconds,
            "sample_rate": item.sample_rate,
            "channels": item.channels,
            "source_codec": item.source_codec,
            "packaged_codec": item.packaged_codec,
            "reencoded": item.reencoded,
            "packaged_sha256": _sha256(Path(item.path)),
        }
        for item in audio
    ]
    reencoded_audio = [item for item in audio_tracks if item["reencoded"]]
    summary: dict[str, Any] = {
        "source_url": _SOURCE_URL,
        "terms_url": _ARCHIVED_TERMS_URL,
        "archive_parts": archive_parts,
        "raw_video_count": len(videos),
        "raw_annotation_groups": len(list(annotations_dir.glob("*.txt"))),
        "raw_annotation_count": raw_annotation_count,
        "excluded_annotations": excluded,
        "annotation_media_name_normalizations": media_name_normalizations,
        "annotation_duration_findings": annotation_duration_findings,
        "transcoded_videos": transcoded_videos,
        "video_codecs": dict(sorted(Counter(video.codec for video in videos).items())),
        "retrieval": retrieval_summary,
        "audio": {
            "missing_video_ids": missing_audio_ids,
            "source_codecs": dict(
                sorted(Counter(item.source_codec for item in audio).items())
            ),
            "packaged_codecs": dict(
                sorted(Counter(item.packaged_codec for item in audio).items())
            ),
            "reencoded_tracks": reencoded_audio,
            "tracks": audio_tracks,
            "retrieval": audio_retrieval_summary,
        },
    }
    card = _dataset_card(summary)
    audio_card = _audio_dataset_card(summary)
    _save_local(work_dir, corpus, queries, qrels, annotation_dataset, card, summary)
    _save_audio_local(
        work_dir,
        audio_corpus,
        audio_queries,
        audio_qrels,
        audio_card,
        summary,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.push:
        revision = _publish(
            args.repo_id,
            corpus,
            queries,
            qrels,
            annotation_dataset,
            card,
            args.upload_workers,
        )
        print(f"Published https://huggingface.co/datasets/{args.repo_id}")
        print(f"Pinned revision: {revision}")
    if args.push_audio:
        audio_revision = _publish_audio(
            args.audio_repo_id,
            audio_corpus,
            audio_queries,
            audio_qrels,
            audio_card,
            summary,
            args.upload_workers,
        )
        print(f"Published https://huggingface.co/datasets/{args.audio_repo_id}")
        print(f"Pinned audio revision: {audio_revision}")
    if not args.push and not args.push_audio:
        print(f"Video export: {work_dir / 'export'}")
        print(f"Audio export: {work_dir / 'audio_export'}")
        print("Re-run with --push and/or --push-audio after reviewing the audits.")


if __name__ == "__main__":
    main()
