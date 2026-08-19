#!/usr/bin/env python3
"""Build Stanford I2V 600K as an MTEB image-to-video+audio retrieval dataset.

The official Stanford release stores the videos in split gzip-compressed tar
archives. Reconstructing the 600K release normally requires roughly 500 GB of
compressed scratch space even though its selected videos occupy about 62 GB.
This script instead uses a bounded prefetch window for verified split parts,
exposes the parts as one continuous stream, and extracts only paths listed in
the official 600K corpus manifest.

The source record does not specify a dataset-wide license. The generated dataset
card records that status as ``not specified`` without asserting rights that the
source does not grant.

Examples:
  # Validate metadata and print the native benchmark counts.
  uv run python scripts/data/stanford_i2v_retrieval/create_data.py \
      --work-dir /tmp/stanford_i2v --metadata-only

  # Reconstruct all official media, validate it, and save local Arrow datasets.
  uv run python scripts/data/stanford_i2v_retrieval/create_data.py \
      --work-dir /path/with/at/least/80GB/free --download-media --save-to-disk

  # Publish the frozen evaluation dataset using the currently authenticated user.
  uv run python scripts/data/stanford_i2v_retrieval/create_data.py \
      --work-dir /path/to/stanford_i2v --download-media --push \
      --repo-id Cerru02/Stanford-I2V-600K
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from datasets import Audio, Dataset, DatasetDict, Image, Video
from huggingface_hub import HfApi, create_repo, get_token

_RECORD_URL = "https://purl.stanford.edu/zx935qw7203.json"
_FILE_BASE_URL = "https://stacks.stanford.edu/file/druid:zx935qw7203"
_SOURCE_RECORD_VERSION = 15
_ARCHIVE_GROUPS = ("201210", "201211", "201212", "201301", "rest_videos_600k_4M")
# This source clip has corrupt AAC packets until second 69. Its only relevance
# annotation is at 2:25-2:27, within the recoverable portion. Preserve the
# timeline by replacing the corrupt prefix with silence and re-encoding the
# decodable suffix.
_AUDIO_REPAIRS = {
    "20130923/Early_Start_20130923_0200_cc_segment/07.700k.mp4": 69.0,
}
_SMALL_FILES = (
    "README.txt",
    "queries.tar.gz",
    "600k_dataset_public.txt",
    "600k_dataset_queries_public.txt",
    "full_dataset_public.txt",
    "full_dataset_queries_public.txt",
    "list_database_clips_600k_public.txt",
    "list_rest_videos_600k_4M.txt",
)


@dataclass(frozen=True)
class SourceFile:
    filename: str
    size: int
    md5: str


def _iter_objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_objects(child)


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_record(work_dir: Path) -> tuple[dict[str, Any], dict[str, SourceFile]]:
    record_path = work_dir / "source" / "stanford_record_v15.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if not record_path.exists():
        with urllib.request.urlopen(_RECORD_URL, timeout=120) as response:
            record_path.write_bytes(response.read())
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("version") != _SOURCE_RECORD_VERSION:
        raise RuntimeError(
            f"Expected Stanford record version {_SOURCE_RECORD_VERSION}, "
            f"found {record.get('version')!r}. Review source changes before rebuilding."
        )

    files: dict[str, SourceFile] = {}
    for obj in _iter_objects(record):
        filename = obj.get("filename")
        size = obj.get("size")
        if not isinstance(filename, str) or not isinstance(size, int):
            continue
        md5 = next(
            (
                item["digest"]
                for item in obj.get("hasMessageDigests", [])
                if item.get("type") == "md5"
            ),
            None,
        )
        if not isinstance(md5, str):
            raise RuntimeError(f"No MD5 digest in source record for {filename}")
        files[filename] = SourceFile(filename=filename, size=size, md5=md5)
    return record, files


def _download_verified(
    source: SourceFile,
    destination: Path,
    *,
    retries: int,
    progress_interval_bytes: int = 1024**3,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size == source.size
            and _md5(destination) == source.md5
        ):
            return destination
        destination.unlink()

    partial = destination.with_name(destination.name + ".part")
    url = f"{_FILE_BASE_URL}/{source.filename}"
    for attempt in range(1, retries + 2):
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded > source.size:
            partial.unlink()
            downloaded = 0
        if downloaded == source.size:
            if _md5(partial) == source.md5:
                partial.replace(destination)
                return destination
            partial.unlink()
            downloaded = 0

        request = urllib.request.Request(url)
        if downloaded:
            request.add_header("Range", f"bytes={downloaded}-")
        next_report = (
            (downloaded // progress_interval_bytes) + 1
        ) * progress_interval_bytes
        started = time.monotonic()
        try:
            print(
                f"{'Resuming' if downloaded else 'Downloading'} {source.filename} "
                f"({source.size / 1024**3:.2f} GiB), attempt {attempt}/{retries + 1}",
                flush=True,
            )
            response = urllib.request.urlopen(request, timeout=120)
            content_range = response.headers.get("Content-Range", "")
            if downloaded and content_range.startswith(f"bytes {downloaded}-"):
                output_mode = "ab"
            else:
                if downloaded:
                    print(
                        f"  {source.filename}: server ignored Range; restarting",
                        flush=True,
                    )
                downloaded = 0
                next_report = progress_interval_bytes
                output_mode = "wb"
            attempt_start = downloaded

            with response, partial.open(output_mode) as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        elapsed = max(time.monotonic() - started, 0.001)
                        transferred = downloaded - attempt_start
                        print(
                            f"  {source.filename}: {downloaded / 1024**3:.1f}/"
                            f"{source.size / 1024**3:.1f} GiB "
                            f"({transferred / 1024**2 / elapsed:.1f} MiB/s)",
                            flush=True,
                        )
                        next_report += progress_interval_bytes
            if downloaded != source.size:
                raise RuntimeError(
                    f"size mismatch: expected {source.size}, downloaded {downloaded}"
                )
            digest = _md5(partial)
            if digest != source.md5:
                raise RuntimeError(
                    f"MD5 mismatch: expected {source.md5}, calculated {digest}"
                )
            partial.replace(destination)
            return destination
        except Exception as error:
            if partial.exists() and partial.stat().st_size >= source.size:
                partial.unlink()
            if attempt > retries:
                raise RuntimeError(
                    f"Failed to download {source.filename}: {error}"
                ) from error
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(
                f"  retrying {source.filename} after {error!s} in {delay}s", flush=True
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


class _VerifiedPartStream(io.RawIOBase):
    """Read a split archive continuously with bounded, ordered prefetching."""

    def __init__(
        self,
        parts: Iterable[SourceFile],
        scratch_dir: Path,
        *,
        retries: int,
        download_workers: int,
    ) -> None:
        self._parts = list(parts)
        self._scratch_dir = scratch_dir
        self._retries = retries
        self._download_workers = max(1, download_workers)
        self._executor = ThreadPoolExecutor(max_workers=self._download_workers)
        self._futures: dict[int, Future[Path]] = {}
        self._next_index = 0
        self._current: BinaryIO | None = None
        self._current_path: Path | None = None
        for index in range(min(self._download_workers, len(self._parts))):
            self._schedule(index)

    def readable(self) -> bool:
        return True

    def _schedule(self, index: int) -> None:
        source = self._parts[index]
        self._futures[index] = self._executor.submit(
            _download_verified,
            source,
            self._scratch_dir / source.filename,
            retries=self._retries,
        )

    def _advance(self) -> bool:
        self._discard_current()
        if self._next_index >= len(self._parts):
            return False
        index = self._next_index
        self._next_index += 1
        path = self._futures.pop(index).result()
        next_to_schedule = index + self._download_workers
        if next_to_schedule < len(self._parts):
            self._schedule(next_to_schedule)
        self._current_path = path
        self._current = path.open("rb")
        return True

    def _discard_current(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        if self._current_path is not None:
            self._current_path.unlink(missing_ok=True)
            self._current_path = None

    def readinto(self, buffer: bytearray) -> int:
        while self._current is not None or self._advance():
            assert self._current is not None
            count = self._current.readinto(buffer)
            if count:
                return count
            self._discard_current()
        return 0

    def close(self) -> None:
        self._discard_current()
        for future in self._futures.values():
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        super().close()


def _archive_parts(files: dict[str, SourceFile], group: str) -> list[SourceFile]:
    prefix = f"{group}.tar.gz"
    parts = [
        source
        for name, source in files.items()
        if name.startswith(prefix) and name != prefix and not name.startswith("md5.")
    ]
    parts.sort(key=lambda item: item.filename)
    if not parts:
        raise RuntimeError(f"No source archive parts found for {group}")
    return parts


def _safe_output_path(root: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"Unsafe path in source archive: {member_name}")
    output = root.joinpath(*member_path.parts)
    output.resolve().relative_to(root.resolve())
    return output


def _extract_group(
    group: str,
    parts: list[SourceFile],
    wanted: set[str],
    videos_dir: Path,
    scratch_dir: Path,
    *,
    retries: int,
    download_workers: int,
) -> None:
    missing_before = {name for name in wanted if not (videos_dir / name).is_file()}
    if not missing_before:
        print(
            f"Archive group {group}: all {len(wanted)} selected videos already present"
        )
        return

    print(
        f"Archive group {group}: scanning {sum(p.size for p in parts) / 1024**3:.2f} GiB "
        f"for {len(missing_before)} missing selected videos",
        flush=True,
    )
    raw = _VerifiedPartStream(
        parts,
        scratch_dir,
        retries=retries,
        download_workers=download_workers,
    )
    extracted = 0
    try:
        with (
            io.BufferedReader(raw, buffer_size=8 * 1024 * 1024) as stream,
            tarfile.open(fileobj=stream, mode="r|gz") as archive,
        ):
            for member in archive:
                name = member.name.removeprefix("./")
                if name not in missing_before:
                    continue
                if not member.isfile():
                    raise RuntimeError(
                        f"Expected a regular file for selected member {name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Unable to read selected member {name}")
                output = _safe_output_path(videos_dir, name)
                output.parent.mkdir(parents=True, exist_ok=True)
                partial = output.with_name(output.name + ".part")
                with source, partial.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
                partial.replace(output)
                extracted += 1
                if extracted % 100 == 0:
                    print(
                        f"  archive group {group}: extracted {extracted}/"
                        f"{len(missing_before)}",
                        flush=True,
                    )
    finally:
        raw.close()

    missing_after = sorted(name for name in wanted if not (videos_dir / name).is_file())
    if missing_after:
        preview = "\n".join(missing_after[:20])
        raise RuntimeError(
            f"Archive group {group} is missing {len(missing_after)} manifest videos:\n{preview}"
        )
    print(f"Archive group {group}: extracted {extracted} videos", flush=True)


def _extract_queries(archive_path: Path, queries_dir: Path) -> None:
    queries_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            output = _safe_output_path(queries_dir, member.name.removeprefix("./"))
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read query member {member.name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.name + ".part")
            with source, partial.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            partial.replace(output)


def _read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_query_manifest(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line in _read_nonempty_lines(path):
        query_id, source_path = line.split(maxsplit=1)
        if query_id in queries:
            raise RuntimeError(f"Duplicate query ID in source manifest: {query_id}")
        if not source_path.startswith("queries/"):
            raise RuntimeError(f"Unexpected query path: {source_path}")
        queries[query_id] = source_path.removeprefix("queries/")
    return queries


def _parse_qrels(path: Path) -> tuple[list[tuple[str, str, int]], int]:
    qrels: set[tuple[str, str, int]] = set()
    temporal_annotations = 0
    for line in _read_nonempty_lines(path):
        fields = line.split()
        query_id = fields[0]
        triples = fields[1:]
        if len(triples) % 3:
            raise RuntimeError(f"Malformed ground-truth line for query {query_id}")
        temporal_annotations += len(triples) // 3
        for index in range(0, len(triples), 3):
            qrels.add((query_id, triples[index], 1))
    return sorted(qrels), temporal_annotations


def _image_is_decodable(path: Path) -> bool:
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _video_is_decodable(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate Stanford I2V videos")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    try:
        data = json.loads(probe.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except (ValueError, json.JSONDecodeError):
        return False
    return bool(data.get("streams")) and duration > 0


def _audio_is_decodable(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to validate Stanford I2V audio")
    decode = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return decode.returncode == 0


def _audio_path(audio_dir: Path, video_id: str) -> Path:
    audio_id = str(PurePosixPath(video_id).with_suffix(".m4a"))
    return _safe_output_path(audio_dir, audio_id)


def _extract_audio_track(
    video_path: Path,
    audio_path: Path,
    *,
    repair_start_seconds: float | None = None,
) -> tuple[bool, bool]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract Stanford I2V audio")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    partial = audio_path.with_name(f".{audio_path.stem}.partial{audio_path.suffix}")
    partial.unlink(missing_ok=True)
    if repair_start_seconds is None:
        lossless_command = [
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "copy",
        ]
        remux = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                *lossless_command,
                "-movflags",
                "+faststart",
                str(partial),
            ],
            capture_output=True,
            text=True,
        )
        if remux.returncode == 0 and _audio_is_decodable(partial):
            partial.replace(audio_path)
            return True, False
        partial.unlink(missing_ok=True)

        command = [
            "-fflags",
            "+discardcorrupt",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ]
    else:
        command = [
            "-f",
            "lavfi",
            "-t",
            str(repair_start_seconds),
            "-i",
            "anullsrc=sample_rate=48000:channel_layout=stereo",
            "-ss",
            str(repair_start_seconds),
            "-i",
            str(video_path),
            "-filter_complex",
            "[0:a:0][1:a:0]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ]
    remux = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            *command,
            "-movflags",
            "+faststart",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if remux.returncode != 0 or not _audio_is_decodable(partial):
        partial.unlink(missing_ok=True)
        return False, True
    partial.replace(audio_path)
    return True, True


def _extract_audio_tracks(
    corpus_ids: list[str],
    videos_dir: Path,
    audio_dir: Path,
    *,
    workers: int,
) -> tuple[set[str], set[str]]:
    available = [
        video_id for video_id in corpus_ids if (videos_dir / video_id).is_file()
    ]

    def extract(video_id: str) -> tuple[bool, bool]:
        return _extract_audio_track(
            videos_dir / video_id,
            _audio_path(audio_dir, video_id),
            repair_start_seconds=_AUDIO_REPAIRS.get(video_id),
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(extract, available))
        failed = {
            video_id
            for video_id, (valid, _) in zip(available, results, strict=True)
            if not valid
        }
        reencoded = {
            video_id
            for video_id, (valid, source_is_malformed) in zip(
                available, results, strict=True
            )
            if valid and source_is_malformed
        }
    print(
        f"Audio extraction: available_videos={len(available)} "
        f"reencoded={len(reencoded)} failed={len(failed)}",
        flush=True,
    )
    return failed, reencoded


def _validate_media(
    queries: dict[str, str],
    corpus_ids: list[str],
    queries_dir: Path,
    videos_dir: Path,
    audio_dir: Path,
    *,
    workers: int,
) -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    missing_queries = {
        query_id
        for query_id, path in queries.items()
        if not (queries_dir / path).is_file()
    }
    bad_queries = {
        query_id
        for query_id, path in queries.items()
        if query_id not in missing_queries
        and not _image_is_decodable(queries_dir / path)
    }
    missing_videos = {
        video_id for video_id in corpus_ids if not (videos_dir / video_id).is_file()
    }
    candidates = [video_id for video_id in corpus_ids if video_id not in missing_videos]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        checks = pool.map(
            lambda video_id: _video_is_decodable(videos_dir / video_id), candidates
        )
        bad_videos = {
            video_id
            for video_id, valid in zip(candidates, checks, strict=True)
            if not valid
        }
    audio_candidates = [
        video_id
        for video_id in corpus_ids
        if video_id not in missing_videos and _audio_path(audio_dir, video_id).is_file()
    ]
    missing_audio = set(candidates) - set(audio_candidates)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        checks = pool.map(
            lambda video_id: _audio_is_decodable(_audio_path(audio_dir, video_id)),
            audio_candidates,
        )
        bad_audio = {
            video_id
            for video_id, valid in zip(audio_candidates, checks, strict=True)
            if not valid
        }
    return (
        missing_queries,
        bad_queries,
        missing_videos,
        bad_videos,
        missing_audio,
        bad_audio,
    )


def _relevance_distribution(counts: Counter[str]) -> dict[str, float | int]:
    if not counts:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(counts.values()),
        "mean": sum(counts.values()) / len(counts),
        "max": max(counts.values()),
    }


def _build_datasets(
    queries: dict[str, str],
    corpus_ids: list[str],
    qrels: list[tuple[str, str, int]],
    queries_dir: Path,
    videos_dir: Path,
    audio_dir: Path,
    *,
    unavailable_queries: set[str],
    unavailable_videos: set[str],
    missing_or_corrupt_videos: set[str],
    missing_or_corrupt_audio: set[str],
    reencoded_audio: set[str],
    temporal_annotations: int,
) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    available_corpus = [
        video_id for video_id in corpus_ids if video_id not in unavailable_videos
    ]
    available_corpus_set = set(available_corpus)
    filtered_qrels = [
        item
        for item in qrels
        if item[0] not in unavailable_queries and item[1] in available_corpus_set
    ]
    queries_with_positive = {query_id for query_id, _, _ in filtered_qrels}
    available_queries = [
        query_id
        for query_id in queries
        if query_id not in unavailable_queries and query_id in queries_with_positive
    ]
    available_query_set = set(available_queries)
    filtered_qrels = [item for item in filtered_qrels if item[0] in available_query_set]
    original_qrels_per_query = Counter(query_id for query_id, _, _ in qrels)
    surviving_qrels_per_query = Counter(query_id for query_id, _, _ in filtered_qrels)
    original_positive_videos = {corpus_id for _, corpus_id, _ in qrels}
    surviving_positive_videos = {corpus_id for _, corpus_id, _ in filtered_qrels}

    corpus = Dataset.from_dict(
        {
            "id": available_corpus,
            "video": [str(videos_dir / video_id) for video_id in available_corpus],
            "audio": [
                str(_audio_path(audio_dir, video_id)) for video_id in available_corpus
            ],
        }
    ).cast_column("video", Video())
    corpus = corpus.cast_column("audio", Audio())
    query_dataset = Dataset.from_dict(
        {
            "id": available_queries,
            "image": [
                str(queries_dir / queries[query_id]) for query_id in available_queries
            ],
        }
    ).cast_column("image", Image())
    qrel_dataset = Dataset.from_dict(
        {
            "query-id": [query_id for query_id, _, _ in filtered_qrels],
            "corpus-id": [corpus_id for _, corpus_id, _ in filtered_qrels],
            "score": [score for _, _, score in filtered_qrels],
        }
    )
    summary = {
        "source_record_version": _SOURCE_RECORD_VERSION,
        "original_queries": len(queries),
        "original_corpus": len(corpus_ids),
        "original_qrels": len(qrels),
        "full_release_corpus": 84443,
        "query_manifest_identical_to_full": True,
        "qrel_manifest_identical_to_full": True,
        "original_temporal_annotations": temporal_annotations,
        "original_positive_videos": len(original_positive_videos),
        "original_distractor_videos": len(corpus_ids) - len(original_positive_videos),
        "original_relevant_videos_per_query": _relevance_distribution(
            original_qrels_per_query
        ),
        "surviving_queries": len(available_queries),
        "surviving_corpus": len(available_corpus),
        "surviving_qrels": len(filtered_qrels),
        "surviving_positive_videos": len(surviving_positive_videos),
        "surviving_distractor_videos": len(available_corpus)
        - len(surviving_positive_videos),
        "surviving_relevant_videos_per_query": _relevance_distribution(
            surviving_qrels_per_query
        ),
        "missing_or_corrupt_queries": sorted(unavailable_queries),
        "missing_or_corrupt_videos": sorted(missing_or_corrupt_videos),
        "missing_or_corrupt_audio": sorted(missing_or_corrupt_audio),
        "reencoded_audio": sorted(reencoded_audio & available_corpus_set),
        "repaired_audio": {
            video_id: {"silent_prefix_seconds": start_seconds}
            for video_id, start_seconds in _AUDIO_REPAIRS.items()
            if video_id in available_corpus_set
        },
        "queries_dropped_without_positive": sorted(
            set(queries) - unavailable_queries - queries_with_positive
        ),
    }
    return corpus, query_dataset, qrel_dataset, summary


def _dataset_card(summary: dict[str, Any]) -> str:
    return f"""---
license: unknown
pretty_name: Stanford I2V 600K
task_categories:
- image-to-video
tags:
- mteb
- moeb
- image-to-video-audio-retrieval
---

# Stanford I2V 600K

Frozen MTEB/MOEB representation of the official Stanford I2V 600K evaluation
release for image-to-video+audio retrieval.

## Construction

The dataset is reconstructed from Stanford Digital Repository record
[`zx935qw7203`](https://purl.stanford.edu/zx935qw7203), version
`{_SOURCE_RECORD_VERSION}`. Query IDs, corpus paths, and relevance annotations
are preserved from `600k_dataset_queries_public.txt`,
`list_database_clips_600k_public.txt`, and `600k_dataset_public.txt`.
Multiple temporal annotations for the same query/video pair are collapsed to one
binary scene-level relevance judgment, matching the source scene-retrieval
protocol.

The construction script is maintained in the MTEB repository at
`scripts/data/stanford_i2v_retrieval/create_data.py`. Every source archive part is checked
against the size and MD5 digest in the Stanford record before extraction. Images
are decoded, videos are probed for a valid video stream, and their AAC soundtracks
are placed in a separate audio column and fully decoded with strict error handling.
Soundtracks with valid AAC streams are losslessly remuxed. The
{len(summary["reencoded_audio"])} tracks containing malformed source packets are
decoded and re-encoded while preserving timestamp gaps. One of those tracks has
an unrecoverable first 69 seconds, which are replaced with silence; its only
relevance annotation (2:25-2:27) lies in the recoverable suffix.

## Frozen evaluation contents

- Queries: {summary["surviving_queries"]} (official: {summary["original_queries"]})
- Corpus video+audio clips: {summary["surviving_corpus"]} (official: {summary["original_corpus"]})
- Binary query/video qrels: {summary["surviving_qrels"]} (official: {summary["original_qrels"]})
- Source temporal annotations: {summary["original_temporal_annotations"]}
- Distinct positive corpus videos: {summary["surviving_positive_videos"]} (official: {summary["original_positive_videos"]})
- Distractor corpus videos: {summary["surviving_distractor_videos"]} (official: {summary["original_distractor_videos"]})
- Missing or corrupt queries: {len(summary["missing_or_corrupt_queries"])}
- Missing or corrupt videos: {len(summary["missing_or_corrupt_videos"])}
- Missing or corrupt audio tracks: {len(summary["missing_or_corrupt_audio"])}
- Re-encoded source-malformed audio tracks: {len(summary["reencoded_audio"])}
- Tracks with an unrecoverable prefix replaced by silence: {len(summary["repaired_audio"])}
- Queries removed because no positive survived: {len(summary["queries_dropped_without_positive"])}

Configs follow MTEB's standard retrieval representation: `queries`, `corpus`,
and `qrels`, each with a `test` split.

## Evaluation

The source benchmark reports mean average precision over the top 100 retrieved
scenes. The corresponding MTEB primary metric is `map_at_100`. Stanford's scorer
divides each query's precision sum by the total number of relevant corpus videos,
which matches standard truncated AP@100 for this release.

## Comparability with the 2015 paper

The 600K query and ground-truth manifests are byte-identical to the full release:
all 229 queries and all annotated positive videos are preserved. The corpus is
the official 3,401-video 600K corpus rather than the 84,443-video full corpus used
for the principal 2015 paper results. The query set, qrels, and metric are
therefore identical, but scores are not directly comparable because the 600K
release contains fewer distractor videos. The 600K release was documented in the
later Bloom-filter retrieval paper.

## Licensing and provenance

The Stanford source record provides public download access but does not specify a
dataset-wide license. Accordingly, this repository records the license as **not
specified** and does not claim that the source media is under an open license.
Query images originate from news websites and corpus videos are recorded
newscasts; rights may remain with their respective owners. Users are responsible
for confirming that their use complies with applicable source terms and law.

## Citation

```bibtex
@inproceedings{{AraujoMMSYS2015,
  author = {{Araujo, A. and Chaves, J. and Chen, D. and Angst, R. and Girod, B.}},
  booktitle = {{Proc. ACM Multimedia Systems}},
  title = {{Stanford I2V: A News Video Dataset for Query-by-Image Experiments}},
  year = {{2015}},
}}

@article{{AraujoArxiv2016,
  author = {{Araujo, A. and Chaves, J. and Lakshman, H. and Angst, R. and Girod, B.}},
  journal = {{arXiv preprint arXiv:1604.07939}},
  title = {{Large-Scale Query-by-Image Video Retrieval Using Bloom Filters}},
  year = {{2016}},
}}
```
"""


def _publish(
    repo_id: str,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    summary: dict[str, Any],
    work_dir: Path,
) -> str:
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found; run `hf auth login` before --push"
        )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=_dataset_card(summary).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add Stanford I2V dataset card",
    )
    DatasetDict({"test": corpus}).push_to_hub(
        repo_id,
        "corpus",
        token=token,
        max_shard_size="500MB",
        commit_message="Add Stanford I2V video+audio corpus",
    )
    DatasetDict({"test": queries}).push_to_hub(
        repo_id,
        "queries",
        token=token,
        commit_message="Add Stanford I2V image queries",
    )
    DatasetDict({"test": qrels}).push_to_hub(
        repo_id,
        "qrels",
        token=token,
        commit_message="Add Stanford I2V relevance judgments",
    )
    sha = api.dataset_info(repo_id).sha
    (work_dir / "hub_revision.txt").write_text(f"{sha}\n", encoding="utf-8")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/stanford_i2v"))
    parser.add_argument("--repo-id", default="Cerru02/Stanford-I2V-600K")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--save-to-disk", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Maximum official archive parts prefetched concurrently (default: 4).",
    )
    parser.add_argument("--verify-workers", type=int, default=8)
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=4,
        help="Maximum audio tracks remuxed concurrently (default: 4).",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    source_dir = work_dir / "source"
    metadata_dir = source_dir / "metadata"
    scratch_dir = source_dir / "scratch"
    queries_dir = source_dir / "queries"
    videos_dir = source_dir / "videos"
    audio_dir = source_dir / "audio"
    for directory in (metadata_dir, scratch_dir, queries_dir, videos_dir, audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    record, files = _load_source_record(work_dir)
    access = record.get("access", {})
    license_fields = [
        item
        for item in _iter_objects(record)
        if any("license" in key.lower() for key in item)
    ]
    if license_fields:
        raise RuntimeError(
            "Source record now contains license metadata; review it before rebuilding"
        )
    if access.get("download") != "world":
        raise RuntimeError(
            f"Unexpected source download access: {access.get('download')!r}"
        )

    for filename in _SMALL_FILES:
        source = files.get(filename)
        if source is None:
            raise RuntimeError(
                f"Required source file is absent from record: {filename}"
            )
        _download_verified(
            source,
            metadata_dir / filename,
            retries=args.download_retries,
            progress_interval_bytes=64 * 1024 * 1024,
        )

    _extract_queries(metadata_dir / "queries.tar.gz", queries_dir)
    queries = _parse_query_manifest(metadata_dir / "600k_dataset_queries_public.txt")
    corpus_ids = _read_nonempty_lines(
        metadata_dir / "list_database_clips_600k_public.txt"
    )
    qrels, temporal_annotations = _parse_qrels(metadata_dir / "600k_dataset_public.txt")
    if (metadata_dir / "600k_dataset_queries_public.txt").read_bytes() != (
        metadata_dir / "full_dataset_queries_public.txt"
    ).read_bytes():
        raise RuntimeError("600K and full query manifests are no longer identical")
    if (metadata_dir / "600k_dataset_public.txt").read_bytes() != (
        metadata_dir / "full_dataset_public.txt"
    ).read_bytes():
        raise RuntimeError(
            "600K and full ground-truth manifests are no longer identical"
        )
    rest_paths = set(
        _read_nonempty_lines(metadata_dir / "list_rest_videos_600k_4M.txt")
    )

    if len(queries) != 229 or len(corpus_ids) != 3401 or len(qrels) != 1280:
        raise RuntimeError(
            "Unexpected native counts: "
            f"queries={len(queries)}, corpus={len(corpus_ids)}, qrels={len(qrels)}"
        )
    if len(set(corpus_ids)) != len(corpus_ids):
        raise RuntimeError("Duplicate corpus IDs in official 600K manifest")
    query_ids = set(queries)
    corpus_id_set = set(corpus_ids)
    if any(
        query_id not in query_ids or corpus_id not in corpus_id_set
        for query_id, corpus_id, _ in qrels
    ):
        raise RuntimeError(
            "Official qrels reference IDs outside the query/corpus manifests"
        )

    print(
        f"Native Stanford I2V 600K: queries={len(queries)} "
        f"corpus={len(corpus_ids)} qrels={len(qrels)}",
        flush=True,
    )
    if args.metadata_only:
        return

    if args.download_media:
        paths_by_group: dict[str, set[str]] = {
            group: set() for group in _ARCHIVE_GROUPS
        }
        for video_id in corpus_ids:
            group = "rest_videos_600k_4M" if video_id in rest_paths else video_id[:6]
            if group not in paths_by_group:
                raise RuntimeError(
                    f"No archive group mapped for corpus video {video_id}"
                )
            paths_by_group[group].add(video_id)
        for group in _ARCHIVE_GROUPS:
            _extract_group(
                group,
                _archive_parts(files, group),
                paths_by_group[group],
                videos_dir,
                scratch_dir,
                retries=args.download_retries,
                download_workers=args.download_workers,
            )

    _, reencoded_audio = _extract_audio_tracks(
        corpus_ids,
        videos_dir,
        audio_dir,
        workers=args.audio_workers,
    )
    (
        missing_queries,
        bad_queries,
        missing_videos,
        bad_videos,
        missing_audio,
        bad_audio,
    ) = _validate_media(
        queries,
        corpus_ids,
        queries_dir,
        videos_dir,
        audio_dir,
        workers=args.verify_workers,
    )
    unavailable_queries = missing_queries | bad_queries
    video_failures = missing_videos | bad_videos
    audio_failures = missing_audio | bad_audio
    unavailable_videos = video_failures | audio_failures
    if (unavailable_queries or unavailable_videos) and not args.allow_missing:
        raise RuntimeError(
            "Media validation failed. Re-run with --download-media to fetch missing files, "
            "or --allow-missing to freeze a deterministic surviving benchmark. "
            f"missing_queries={len(missing_queries)} bad_queries={len(bad_queries)} "
            f"missing_videos={len(missing_videos)} bad_videos={len(bad_videos)} "
            f"missing_audio={len(missing_audio)} bad_audio={len(bad_audio)}"
        )

    corpus, query_dataset, qrel_dataset, summary = _build_datasets(
        queries,
        corpus_ids,
        qrels,
        queries_dir,
        videos_dir,
        audio_dir,
        unavailable_queries=unavailable_queries,
        unavailable_videos=unavailable_videos,
        missing_or_corrupt_videos=video_failures,
        missing_or_corrupt_audio=audio_failures,
        reencoded_audio=reencoded_audio,
        temporal_annotations=temporal_annotations,
    )
    summary_path = work_dir / "construction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.save_to_disk:
        export_dir = work_dir / "mteb_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        corpus.save_to_disk(export_dir / "corpus")
        query_dataset.save_to_disk(export_dir / "queries")
        qrel_dataset.save_to_disk(export_dir / "qrels")
        print(f"Saved local datasets to {export_dir}", flush=True)

    if args.push:
        sha = _publish(
            args.repo_id,
            corpus,
            query_dataset,
            qrel_dataset,
            summary,
            work_dir,
        )
        print(f"Pushed {args.repo_id} at immutable revision {sha}", flush=True)
    elif not args.save_to_disk:
        print(
            "Validation complete; pass --save-to-disk and/or --push to materialize output"
        )


if __name__ == "__main__":
    main()
