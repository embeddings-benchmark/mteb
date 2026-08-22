#!/usr/bin/env python3
"""Build the VCDB core set as an MTEB video-to-video retrieval dataset.

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

from datasets import Dataset, DatasetDict, Value, Video
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
        help="Publish publicly to Hugging Face; requires an authenticated account",
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
    annotations, excluded, media_name_normalizations, raw_annotation_count = (
        _load_annotations(annotations_dir, videos)
    )
    annotation_duration_findings = _audit_annotation_durations(annotations, videos)
    corpus, queries, qrels, annotation_dataset, retrieval_summary = _build_datasets(
        videos, annotations
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
    }
    card = _dataset_card(summary)
    _save_local(work_dir, corpus, queries, qrels, annotation_dataset, card, summary)

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
    else:
        print(f"Local export: {work_dir / 'export'}")
        print("Re-run with --push to publish after reviewing the audit.")


if __name__ == "__main__":
    main()
