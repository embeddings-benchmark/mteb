#!/usr/bin/env python3
"""Build the Lombard GRID image-to-video+audio speaker-retrieval dataset.

The script is reproducible, resumable, and non-publishing by default. It verifies
the official Zenodo sizes and MD5 digests, safely extracts the archives, audits
the source release, selects a balanced fixed protocol, materializes query images,
and validates every selected output with the media stack used by MTEB.

Examples:
  # Download and verify the four required source archives.
  python scripts/data/lombard_grid_i2va/create_data.py download \
      --work-dir /tmp/lombard-grid-i2va

  # Reconcile metadata and media, including a complete FFmpeg decode audit.
  python scripts/data/lombard_grid_i2va/create_data.py inspect \
      --work-dir /tmp/lombard-grid-i2va

  # Build and validate the local queries/corpus/qrels datasets.
  python scripts/data/lombard_grid_i2va/create_data.py build \
      --work-dir /tmp/lombard-grid-i2va

  # Re-run validation without rebuilding media.
  python scripts/data/lombard_grid_i2va/create_data.py validate \
      --work-dir /tmp/lombard-grid-i2va

  # Publishing is a separate, explicit action.
  python scripts/data/lombard_grid_i2va/create_data.py push \
      --work-dir /tmp/lombard-grid-i2va \
      --repo-id Cerru02/LombardGrid-I2VA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import statistics
import subprocess
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Image, Video, load_from_disk
from huggingface_hub import HfApi, create_repo, get_token

_ZENODO_RECORD = "https://zenodo.org/records/3736465"
_ZENODO_API = "https://zenodo.org/api/records/3736465/files"
_PAPER = "https://doi.org/10.1121/1.5042758"
_SELECTION_SEED = "mteb-lombard-grid-i2va-v1"
_SPEAKERS = tuple(f"s{index}" for index in range(2, 56))
_CONDITIONS = ("p", "l")
_QUERY_PER_CONDITION = 5
_CORPUS_PER_CONDITION = 10
_EXPECTED_CORRUPT_FRONT = {
    "s32_l_pwip9p",
    "s32_p_bwwj2n",
    "s33_l_pwajza",
    "s33_p_sgwq2s",
}


@dataclass(frozen=True)
class SourceArchive:
    filename: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"{_ZENODO_API}/{self.filename}/content"


_SOURCE_ARCHIVES = (
    SourceArchive(
        "lombardgrid_json.zip",
        64_938,
        "070bf2b1570d9381215e7e57d8f58a21",
    ),
    SourceArchive(
        "lombardgrid_audio.zip",
        652_614_041,
        "fa0cfc739705323b53ba50e148b3a144",
    ),
    SourceArchive(
        "lombardgrid_front.zip",
        837_239_327,
        "63b546f53267ae3f4cffe0c772317ce1",
    ),
    SourceArchive(
        "lombardgrid_side.zip",
        992_617_582,
        "7fdc4b94f1b04de896e890f4ade355e0",
    ),
)


@dataclass(frozen=True)
class Recording:
    stem: str
    speaker: str
    condition: str
    utterance_code: str
    status: str
    actual_transcription: str | None = None
    legacy_filename: bool = False

    @property
    def canonical_key(self) -> tuple[str, str, str]:
        return (self.speaker, self.condition, self.utterance_code)


@dataclass(frozen=True)
class SelectedRecording:
    id: str
    role: str
    stem: str
    speaker: str
    condition: str
    utterance_code: str


_CORRECT_FILENAME = re.compile(
    r"^(?P<speaker>s[0-9]+)_(?P<condition>[lp])_"
    r"(?P<utterance>[a-z0-9]{6})$"
)
_WRONG_FILENAME = re.compile(
    r"^(?P<speaker>s[0-9]+)_(?P<condition>[lp])_"
    r"(?P<utterance>[a-z0-9]+)_WRONG_(?P<actual>.+)$"
)
_LEGACY_REFERENCE_FILENAME = re.compile(
    r"^(?P<speaker>s[0-9]+)_[0-9]+_[0-9]+_[0-9]+_r_"
    r"(?P<utterance>[a-z0-9]{6})$"
)


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_fingerprint() -> dict[str, dict[str, int | str]]:
    return {
        source.filename: {"size": source.size, "md5": source.md5}
        for source in _SOURCE_ARCHIVES
    }


def _download_verified(
    source: SourceArchive,
    destination: Path,
    *,
    retries: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if (
            destination.stat().st_size == source.size
            and _md5(destination) == source.md5
        ):
            print(f"Verified cached {source.filename}", flush=True)
            return destination
        raise RuntimeError(
            f"Existing {destination} does not match the official size and MD5. "
            "Move it aside before retrying."
        )

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 2):
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded > source.size:
            partial.unlink()
            downloaded = 0
        request = urllib.request.Request(source.url)
        if downloaded:
            request.add_header("Range", f"bytes={downloaded}-")
        try:
            print(
                f"{'Resuming' if downloaded else 'Downloading'} {source.filename} "
                f"({source.size / 1024**3:.3f} GiB), attempt "
                f"{attempt}/{retries + 1}",
                flush=True,
            )
            started = time.monotonic()
            response = urllib.request.urlopen(request, timeout=120)
            content_range = response.headers.get("Content-Range", "")
            append = downloaded > 0 and content_range.startswith(f"bytes {downloaded}-")
            if downloaded and not append:
                downloaded = 0
            with response, partial.open("ab" if append else "wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
            if downloaded != source.size:
                raise RuntimeError(
                    f"Expected {source.size} bytes, downloaded {downloaded}"
                )
            digest = _md5(partial)
            if digest != source.md5:
                raise RuntimeError(f"Expected MD5 {source.md5}, calculated {digest}")
            partial.replace(destination)
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"Verified {source.filename} in {elapsed:.1f}s "
                f"({source.size / 1024**2 / elapsed:.1f} MiB/s)",
                flush=True,
            )
            return destination
        except Exception as error:
            if attempt > retries:
                raise RuntimeError(
                    f"Failed to download {source.filename}: {error}"
                ) from error
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(f"Retrying after {error!s} in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _ensure_sources(
    source_dir: Path,
    *,
    download: bool,
    retries: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for source in _SOURCE_ARCHIVES:
        path = source_dir / source.filename
        if not path.is_file():
            if not download:
                raise RuntimeError(
                    f"Missing {path}. Run the download action or pass --download."
                )
            _download_verified(source, path, retries=retries)
        size = path.stat().st_size
        digest = _md5(path)
        if size != source.size or digest != source.md5:
            raise RuntimeError(
                f"Source mismatch for {source.filename}: expected "
                f"{source.size} bytes/{source.md5}, found {size} bytes/{digest}"
            )
        paths[source.filename] = path
    return paths


def _safe_output_path(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts or "\\" in member_name:
        raise RuntimeError(f"Unsafe path in source archive: {member_name}")
    output = root.joinpath(*member.parts)
    output.resolve().relative_to(root.resolve())
    return output


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _ensure_extracted(
    archives: dict[str, Path],
    extraction_root: Path,
) -> dict[str, Any]:
    marker_path = extraction_root / ".verified_extraction.json"
    fingerprint = _archive_fingerprint()
    marker: dict[str, Any] = {}
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))

    extraction_root.mkdir(parents=True, exist_ok=True)
    archive_summary: dict[str, Any] = {}
    all_complete = marker.get("archives") == fingerprint
    for filename, archive_path in archives.items():
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicate member paths in {filename}")
            if any(_zip_member_is_symlink(info) for info in infos):
                raise RuntimeError(f"Symbolic link found in {filename}")
            for info in infos:
                _safe_output_path(extraction_root, info.filename)

            complete = all_complete and all(
                info.is_dir()
                or (
                    (extraction_root / PurePosixPath(info.filename)).is_file()
                    and (extraction_root / PurePosixPath(info.filename)).stat().st_size
                    == info.file_size
                )
                for info in infos
            )
            if not complete:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise RuntimeError(
                        f"CRC failure in {filename}, member {bad_member}"
                    )
                for info in infos:
                    output = _safe_output_path(extraction_root, info.filename)
                    if info.is_dir():
                        output.mkdir(parents=True, exist_ok=True)
                        continue
                    if output.is_file() and output.stat().st_size == info.file_size:
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    partial = output.with_name(output.name + ".part")
                    with archive.open(info) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    if partial.stat().st_size != info.file_size:
                        raise RuntimeError(f"Incomplete extraction of {info.filename}")
                    partial.replace(output)

            files = [info for info in infos if not info.is_dir()]
            archive_summary[filename] = {
                "files": len(files),
                "directories": len(infos) - len(files),
                "compressed_bytes": sum(info.compress_size for info in files),
                "uncompressed_bytes": sum(info.file_size for info in files),
            }

    marker_path.write_text(
        json.dumps({"archives": fingerprint}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive_summary


def _parse_recording(stem: str) -> Recording:
    match = _CORRECT_FILENAME.fullmatch(stem)
    if match:
        return Recording(
            stem=stem,
            speaker=match.group("speaker"),
            condition=match.group("condition"),
            utterance_code=match.group("utterance"),
            status="CORRECT",
        )
    match = _WRONG_FILENAME.fullmatch(stem)
    if match:
        return Recording(
            stem=stem,
            speaker=match.group("speaker"),
            condition=match.group("condition"),
            utterance_code=match.group("utterance"),
            status="WRONG",
            actual_transcription=match.group("actual"),
        )
    match = _LEGACY_REFERENCE_FILENAME.fullmatch(stem)
    if match:
        return Recording(
            stem=stem,
            speaker=match.group("speaker"),
            condition="p",
            utterance_code=match.group("utterance"),
            status="CORRECT",
            legacy_filename=True,
        )
    raise RuntimeError(f"Unrecognized source filename: {stem}")


def _media_stems(media_root: Path, kind: str) -> set[str]:
    extension = ".wav" if kind == "audio" else ".mov"
    return {
        path.stem
        for path in (media_root / kind).iterdir()
        if path.is_file() and path.suffix.lower() == extension
    }


def _load_metadata_rows(metadata_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    files = sorted(
        (metadata_root / "lombardgrid" / "json").glob("s*.json"),
        key=lambda path: int(path.stem[1:]),
    )
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise RuntimeError(f"Expected a list in {path}")
        rows.extend(value)
    return rows


def _speaker_condition_summary(
    keys: list[tuple[str, str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speaker in _SPEAKERS:
        plain = {
            code for spk, condition, code in keys if spk == speaker and condition == "p"
        }
        lombard = {
            code for spk, condition, code in keys if spk == speaker and condition == "l"
        }
        entries = [key for key in keys if key[0] == speaker]
        result[speaker] = {
            "rows_or_files": len(entries),
            "plain_rows_or_files": sum(key[1] == "p" for key in entries),
            "lombard_rows_or_files": sum(key[1] == "l" for key in entries),
            "plain_unique_codes": len(plain),
            "lombard_unique_codes": len(lombard),
            "intersection": len(plain & lombard),
            "plain_only": sorted(plain - lombard),
            "lombard_only": sorted(lombard - plain),
        }
    return result


def _source_reconciliation(
    media_root: Path,
    metadata_rows: list[dict[str, str]],
) -> tuple[list[Recording], dict[str, Any]]:
    modality_stems = {
        kind: _media_stems(media_root, kind) for kind in ("audio", "front", "side")
    }
    if not (
        modality_stems["audio"] == modality_stems["front"] == modality_stems["side"]
    ):
        raise RuntimeError("Audio, frontal-video, and side-video filenames differ")

    recordings = sorted(
        (_parse_recording(stem) for stem in modality_stems["audio"]),
        key=lambda item: (int(item.speaker[1:]), item.condition, item.stem),
    )
    media_keys = [recording.canonical_key for recording in recordings]
    metadata_keys = [
        (row["SPKR"], row["COND"], row["UTTERANCE"]) for row in metadata_rows
    ]
    media_counter = Counter(media_keys)
    metadata_counter = Counter(metadata_keys)
    summary = {
        "media_files_per_modality": len(recordings),
        "modality_filename_sets_equal": True,
        "metadata_rows": len(metadata_rows),
        "metadata_unique_recording_keys": len(metadata_counter),
        "media_unique_recording_keys": len(media_counter),
        "media_only_recording_keys": [
            "_".join(key)
            for key in sorted(media_counter.keys() - metadata_counter.keys())
        ],
        "metadata_only_recording_keys": [
            "_".join(key)
            for key in sorted(metadata_counter.keys() - media_counter.keys())
        ],
        "metadata_duplicate_recording_keys": {
            "_".join(key): count for key, count in metadata_counter.items() if count > 1
        },
        "media_duplicate_recording_keys": {
            "_".join(key): count for key, count in media_counter.items() if count > 1
        },
        "media_status_counts": dict(Counter(item.status for item in recordings)),
        "legacy_filename_count": sum(item.legacy_filename for item in recordings),
        "metadata_status_counts": dict(Counter(row["STATUS"] for row in metadata_rows)),
        "metadata_by_speaker": _speaker_condition_summary(metadata_keys),
        "media_by_speaker": _speaker_condition_summary(media_keys),
    }
    return recordings, summary


def _probe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream="
            "codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,"
            "sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        return None, process.stderr.strip()
    try:
        return json.loads(process.stdout), None
    except json.JSONDecodeError as error:
        return None, repr(error)


def _decode_with_ffmpeg(path: Path) -> str | None:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return process.stderr.strip() if process.returncode else None


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": max(values),
        "mean": statistics.fmean(values),
        "sum": sum(values),
    }


def _source_media_audit(
    media_root: Path,
    reports_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    report_path = reports_dir / "source_media_audit.json"
    fingerprint = _archive_fingerprint()
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("archive_fingerprint") == fingerprint:
            print("Using cached complete source-media audit", flush=True)
            return cached

    paths = [
        path
        for kind in ("audio", "front", "side")
        for path in sorted((media_root / kind).iterdir())
        if path.is_file()
    ]

    def inspect(path: Path) -> tuple[Path, dict[str, Any] | None, str | None]:
        probe, probe_error = _probe(path)
        if probe_error is not None:
            return path, None, probe_error
        decode_error = _decode_with_ffmpeg(path)
        return path, probe, decode_error

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(inspect, paths))

    errors: dict[str, str] = {}
    valid: dict[str, dict[str, Any]] = {}
    for path, probe, error in results:
        relative = str(path.relative_to(media_root))
        if error is not None:
            errors[relative] = error
        elif probe is not None:
            valid[relative] = probe

    corrupt_front = {
        Path(relative).stem
        for relative in errors
        if PurePosixPath(relative).parts[0] == "front"
    }
    if corrupt_front != _EXPECTED_CORRUPT_FRONT or len(errors) != len(corrupt_front):
        raise RuntimeError(
            "Source decode failures changed. Expected only "
            f"{sorted(_EXPECTED_CORRUPT_FRONT)}, found {errors}"
        )

    kinds: dict[str, Any] = {}
    durations: dict[str, dict[str, float]] = {}
    for kind in ("audio", "front", "side"):
        items = {
            relative: data
            for relative, data in valid.items()
            if PurePosixPath(relative).parts[0] == kind
        }
        signatures: Counter[str] = Counter()
        duration_values: list[float] = []
        durations[kind] = {}
        for relative, data in items.items():
            streams = tuple(
                sorted(
                    (
                        stream.get("codec_type"),
                        stream.get("codec_name"),
                        stream.get("width"),
                        stream.get("height"),
                        stream.get("pix_fmt"),
                        stream.get("avg_frame_rate"),
                        stream.get("sample_rate"),
                        stream.get("channels"),
                    )
                    for stream in data.get("streams", [])
                )
            )
            signatures[repr(streams)] += 1
            duration = float(data["format"]["duration"])
            duration_values.append(duration)
            durations[kind][Path(relative).stem] = duration
        kinds[kind] = {
            "decoded_files": len(items),
            "stream_signatures": dict(signatures),
            "duration_seconds": _distribution(duration_values),
        }

    duration_differences: dict[str, Any] = {}
    for video_kind in ("front", "side"):
        common = durations[video_kind].keys() & durations["audio"].keys()
        differences = [
            abs(durations[video_kind][stem] - durations["audio"][stem])
            for stem in common
        ]
        duration_differences[f"{video_kind}_audio"] = {
            "compared": len(common),
            "missing_pairs": len(durations["audio"].keys() - common),
            **_distribution(differences),
            "over_0_1_seconds": sum(value > 0.1 for value in differences),
            "over_0_25_seconds": sum(value > 0.25 for value in differences),
        }

    report = {
        "archive_fingerprint": fingerprint,
        "files_checked": len(paths),
        "files_decoded": len(valid),
        "decode_errors": errors,
        "codec_and_duration_summary": kinds,
        "paired_duration_absolute_differences_seconds": duration_differences,
        "elapsed_seconds": time.monotonic() - started,
        "signal_level_synchronization_independently_verified": False,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _stable_order_key(role: str, stem: str) -> bytes:
    value = f"{_SELECTION_SEED}\0{role}\0{stem}".encode()
    return hashlib.sha256(value).digest()


def _opaque_id(role: str, stem: str) -> str:
    digest = hashlib.sha256(f"{role}\0{stem}".encode()).hexdigest()[:16]
    return f"{'q' if role == 'query' else 'c'}_{digest}"


def _select_protocol(
    recordings: list[Recording],
    unavailable_stems: set[str],
) -> tuple[list[SelectedRecording], list[dict[str, str]]]:
    candidates = [
        recording
        for recording in recordings
        if recording.status == "CORRECT" and not recording.legacy_filename
    ]
    selected: list[SelectedRecording] = []
    fallbacks: list[dict[str, str]] = []
    for speaker in _SPEAKERS:
        used_codes: set[str] = set()
        for role, quota in (
            ("query", _QUERY_PER_CONDITION),
            ("corpus", _CORPUS_PER_CONDITION),
        ):
            for condition in _CONDITIONS:
                ordered = sorted(
                    (
                        recording
                        for recording in candidates
                        if recording.speaker == speaker
                        and recording.condition == condition
                    ),
                    key=lambda item: _stable_order_key(role, item.stem),
                )
                chosen: list[Recording] = []
                for recording in ordered:
                    if recording.utterance_code in used_codes:
                        continue
                    if recording.stem in unavailable_stems:
                        fallbacks.append(
                            {
                                "speaker": speaker,
                                "role": role,
                                "condition": condition,
                                "rejected_stem": recording.stem,
                                "reason": "missing or corrupt required source modality",
                            }
                        )
                        continue
                    chosen.append(recording)
                    used_codes.add(recording.utterance_code)
                    if len(chosen) == quota:
                        break
                if len(chosen) != quota:
                    raise RuntimeError(
                        f"Unable to select {quota} {role}/{condition} recordings "
                        f"for {speaker}; found {len(chosen)}"
                    )
                selected.extend(
                    SelectedRecording(
                        id=_opaque_id(role, recording.stem),
                        role=role,
                        stem=recording.stem,
                        speaker=recording.speaker,
                        condition=recording.condition,
                        utterance_code=recording.utterance_code,
                    )
                    for recording in chosen
                )
        if len(used_codes) != 30:
            raise RuntimeError(
                f"Expected 30 distinct utterance codes for {speaker}, "
                f"found {len(used_codes)}"
            )
    return selected, fallbacks


def _copy_resumable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    partial = destination.with_name(destination.name + ".part")
    with source.open("rb") as input_file, partial.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
    partial.replace(destination)


def _image_is_valid(path: Path) -> bool:
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.width > 0 and image.height > 0
    except Exception:
        return False


def _extract_midpoint_image(source: Path, destination: Path) -> None:
    if destination.is_file() and _image_is_valid(destination):
        return
    from PIL import Image as PILImage
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(source, dimension_order="NHWC")
    metadata = decoder.metadata
    midpoint = metadata.begin_stream_seconds + metadata.duration_seconds / 2
    frame = decoder.get_frame_played_at(midpoint).data.cpu().numpy()
    image = PILImage.fromarray(frame, mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    image.save(partial, format="PNG", optimize=False)
    partial.replace(destination)


def _stream_types(path: Path) -> list[str]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(f"Unable to probe {path}: {process.stderr.strip()}")
    data = json.loads(process.stdout)
    return [stream["codec_type"] for stream in data.get("streams", [])]


def _materialize_video(source: Path, destination: Path) -> bool:
    stream_types = _stream_types(source)
    if stream_types == ["video"]:
        _copy_resumable(source, destination)
        return False
    if "video" not in stream_types:
        raise RuntimeError(f"No video stream in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".part" + destination.suffix)
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"Unable to strip soundtrack from {source}: {process.stderr.strip()}"
        )
    partial.replace(destination)
    return True


def _output_paths(
    output_dir: Path,
    selected: list[SelectedRecording],
) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for item in selected:
        if item.role == "query":
            paths[item.id] = {
                "image": output_dir / "media" / "queries" / f"{item.id}.png"
            }
        else:
            paths[item.id] = {
                "video": output_dir / "media" / "corpus" / "video" / f"{item.id}.mov",
                "audio": output_dir / "media" / "corpus" / "audio" / f"{item.id}.wav",
            }
    return paths


def _materialize_media(
    media_root: Path,
    output_dir: Path,
    selected: list[SelectedRecording],
    *,
    workers: int,
) -> tuple[dict[str, dict[str, Path]], list[str]]:
    paths = _output_paths(output_dir, selected)
    queries = [item for item in selected if item.role == "query"]
    corpus = [item for item in selected if item.role == "corpus"]

    def extract_query(item: SelectedRecording) -> None:
        _extract_midpoint_image(
            media_root / "front" / f"{item.stem}.mov",
            paths[item.id]["image"],
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        list(pool.map(extract_query, queries))

    remuxed: list[str] = []
    for item in corpus:
        if _materialize_video(
            media_root / "side" / f"{item.stem}.mov",
            paths[item.id]["video"],
        ):
            remuxed.append(item.id)
        _copy_resumable(
            media_root / "audio" / f"{item.stem}.wav",
            paths[item.id]["audio"],
        )
    return paths, remuxed


def _validate_output_media(
    selected: list[SelectedRecording],
    paths: dict[str, dict[str, Path]],
    *,
    workers: int,
) -> dict[str, Any]:
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    queries = [item for item in selected if item.role == "query"]
    corpus = [item for item in selected if item.role == "corpus"]

    query_results: list[tuple[str, str]] = []
    for item in queries:
        path = paths[item.id]["image"]
        if not _image_is_valid(path):
            raise RuntimeError(f"Invalid query image: {path}")
        query_results.append((item.id, _sha256(path)))

    def validate_corpus(item: SelectedRecording) -> dict[str, Any]:
        video_path = paths[item.id]["video"]
        audio_path = paths[item.id]["audio"]
        if _stream_types(video_path) != ["video"]:
            raise RuntimeError(f"Corpus video is not visual-only: {video_path}")
        video_decoder = VideoDecoder(video_path, dimension_order="NHWC")
        frames = video_decoder[:]
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise RuntimeError(f"No decoded frames in {video_path}")
        audio_decoder = AudioDecoder(audio_path)
        samples = audio_decoder.get_all_samples().data
        if samples.ndim != 2 or samples.shape[-1] == 0:
            raise RuntimeError(f"No decoded samples in {audio_path}")
        return {
            "id": item.id,
            "video_duration_seconds": video_decoder.metadata.duration_seconds,
            "audio_duration_seconds": audio_decoder.metadata.duration_seconds,
            "video_frames": int(frames.shape[0]),
            "audio_samples": int(samples.shape[-1]),
            "video_sha256": _sha256(video_path),
            "audio_sha256": _sha256(audio_path),
        }

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        corpus_results = list(pool.map(validate_corpus, corpus))

    image_hashes = [digest for _, digest in query_results]
    video_hashes = [item["video_sha256"] for item in corpus_results]
    audio_hashes = [item["audio_sha256"] for item in corpus_results]
    if len(set(image_hashes)) != len(image_hashes):
        raise RuntimeError("Byte-identical query images found")
    if len(set(video_hashes)) != len(video_hashes):
        raise RuntimeError("Byte-identical corpus videos found")
    if len(set(audio_hashes)) != len(audio_hashes):
        raise RuntimeError("Byte-identical corpus audio files found")

    duration_differences = [
        abs(item["video_duration_seconds"] - item["audio_duration_seconds"])
        for item in corpus_results
    ]
    return {
        "query_images_fully_decoded": len(query_results),
        "corpus_videos_fully_decoded_with_torchcodec": len(corpus_results),
        "corpus_audio_fully_decoded_with_torchcodec": len(corpus_results),
        "byte_identical_query_images": 0,
        "byte_identical_corpus_videos": 0,
        "byte_identical_corpus_audio": 0,
        "video_audio_duration_absolute_difference_seconds": _distribution(
            duration_differences
        ),
        "signal_level_synchronization_independently_verified": False,
    }


def _build_qrels(
    selected: list[SelectedRecording],
) -> list[tuple[str, str, int]]:
    queries = [item for item in selected if item.role == "query"]
    corpus = [item for item in selected if item.role == "corpus"]
    return sorted(
        (query.id, document.id, 1)
        for query in queries
        for document in corpus
        if query.speaker == document.speaker
    )


def _validate_protocol(
    selected: list[SelectedRecording],
    qrels: list[tuple[str, str, int]],
) -> dict[str, Any]:
    queries = [item for item in selected if item.role == "query"]
    corpus = [item for item in selected if item.role == "corpus"]
    if len(queries) != 540 or len(corpus) != 1_080 or len(qrels) != 10_800:
        raise RuntimeError(
            f"Unexpected protocol counts: {len(queries)=}, {len(corpus)=}, "
            f"{len(qrels)=}"
        )
    ids = [item.id for item in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate opaque recording IDs")

    query_ids = {item.id for item in queries}
    corpus_ids = {item.id for item in corpus}
    if query_ids & corpus_ids:
        raise RuntimeError("Query/corpus ID overlap")
    if any(
        query_id not in query_ids or corpus_id not in corpus_ids
        for query_id, corpus_id, _ in qrels
    ):
        raise RuntimeError("Qrel references an unknown ID")
    if any(score != 1 for _, _, score in qrels):
        raise RuntimeError("Non-binary qrel score")

    by_query = Counter(query_id for query_id, _, _ in qrels)
    if set(by_query) != query_ids or set(by_query.values()) != {20}:
        raise RuntimeError("Every query must have exactly 20 relevant documents")

    speaker_balance: dict[str, Any] = {}
    for speaker in _SPEAKERS:
        speaker_queries = [item for item in queries if item.speaker == speaker]
        speaker_corpus = [item for item in corpus if item.speaker == speaker]
        if len(speaker_queries) != 10 or len(speaker_corpus) != 20:
            raise RuntimeError(f"Unbalanced speaker {speaker}")
        if Counter(item.condition for item in speaker_queries) != {"p": 5, "l": 5}:
            raise RuntimeError(f"Unbalanced query conditions for {speaker}")
        if Counter(item.condition for item in speaker_corpus) != {"p": 10, "l": 10}:
            raise RuntimeError(f"Unbalanced corpus conditions for {speaker}")
        query_codes = {item.utterance_code for item in speaker_queries}
        corpus_codes = {item.utterance_code for item in speaker_corpus}
        if query_codes & corpus_codes or len(query_codes | corpus_codes) != 30:
            raise RuntimeError(f"Utterance-code leakage for {speaker}")
        speaker_balance[speaker] = {
            "queries": len(speaker_queries),
            "query_plain": 5,
            "query_lombard": 5,
            "corpus": len(speaker_corpus),
            "corpus_plain": 10,
            "corpus_lombard": 10,
            "distinct_utterance_codes": 30,
        }

    return {
        "queries": len(queries),
        "corpus": len(corpus),
        "qrels": len(qrels),
        "relevant_corpus_per_query": 20,
        "speakers": len(_SPEAKERS),
        "speaker_condition_balance": speaker_balance,
        "query_corpus_utterance_code_overlap": 0,
        "published_feature_labels": [],
    }


def _make_datasets(
    selected: list[SelectedRecording],
    qrels: list[tuple[str, str, int]],
    paths: dict[str, dict[str, Path]],
) -> tuple[Dataset, Dataset, Dataset]:
    queries = [item for item in selected if item.role == "query"]
    corpus = [item for item in selected if item.role == "corpus"]
    query_dataset = Dataset.from_dict(
        {
            "id": [item.id for item in queries],
            "image": [str(paths[item.id]["image"]) for item in queries],
        }
    ).cast_column("image", Image())
    corpus_dataset = Dataset.from_dict(
        {
            "id": [item.id for item in corpus],
            "video": [str(paths[item.id]["video"]) for item in corpus],
            "audio": [str(paths[item.id]["audio"]) for item in corpus],
        }
    ).cast_column("video", Video())
    corpus_dataset = corpus_dataset.cast_column("audio", Audio())
    qrel_dataset = Dataset.from_dict(
        {
            "query-id": [query_id for query_id, _, _ in qrels],
            "corpus-id": [corpus_id for _, corpus_id, _ in qrels],
            "score": [score for _, _, score in qrels],
        }
    )
    return corpus_dataset, query_dataset, qrel_dataset


def _feature_schema(dataset: Dataset) -> dict[str, str]:
    return {name: repr(feature) for name, feature in dataset.features.items()}


def _dataset_identity_matches(
    name: str,
    existing: Dataset,
    expected: Dataset,
) -> bool:
    if len(existing) != len(expected) or existing.features != expected.features:
        return False
    identity_columns = (
        ("id",) if name in {"corpus", "queries"} else tuple(expected.column_names)
    )
    return all(
        list(existing[column]) == list(expected[column]) for column in identity_columns
    )


def _save_datasets(
    output_dir: Path,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    *,
    force: bool,
) -> None:
    datasets_dir = output_dir / "datasets"
    for name, dataset in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        destination = datasets_dir / name
        if destination.exists():
            if not force:
                existing = load_from_disk(str(destination))
                if _dataset_identity_matches(name, existing, dataset):
                    print(f"Reusing {destination}", flush=True)
                    continue
                raise RuntimeError(
                    f"Existing generated dataset differs at {destination}; use --force"
                )
            shutil.rmtree(destination)
        dataset.save_to_disk(destination)


def _dataset_card(summary: dict[str, Any]) -> str:
    protocol = summary["protocol_validation"]
    media = summary["selected_media_validation"]
    source = summary["source_reconciliation"]
    fallbacks = summary["selection_fallbacks"]
    remuxed = summary["corpus_videos_remuxed_to_strip_audio"]
    return f"""---
license: cc-by-4.0
pretty_name: Lombard GRID Image-to-Video+Audio Speaker Retrieval
task_categories:
- image-to-video
tags:
- mteb
- moeb
- image-to-video-audio-retrieval
- speaker-retrieval
- biometric-evaluation
---

# Lombard GRID Image-to-Video+Audio Speaker Retrieval

An MTEB/MOEB image-to-video+audio (`i2va`) speaker-retrieval benchmark derived
from the [Lombard GRID corpus]({_ZENODO_RECORD}). Query images show a frontal
view of a talker. Corpus items contain a different utterance from the side/profile
camera together with the matching separate high-quality audio recording. Every
corpus item from the same speaker is relevant.

The source paper introduced the corpus but did not define this retrieval task.
Relevance is derived from native speaker identity labels.

## Frozen evaluation protocol

- Test queries: {protocol["queries"]} (10 per speaker; 5 plain and 5 Lombard)
- Test corpus: {protocol["corpus"]} (20 per speaker; 10 plain and 10 Lombard)
- Binary qrels: {protocol["qrels"]}
- Speakers: {protocol["speakers"]} (`s2` through `s55`; `s1` was excluded by the source)
- Relevant corpus items per query: {protocol["relevant_corpus_per_query"]}
- Distinct utterance codes per speaker: 30
- Query/corpus utterance-code overlap: 0
- Selection seed: `{_SELECTION_SEED}`

Within each speaker and condition, recordings are ordered by SHA-256 over the
fixed seed, role, and source filename. The query and corpus selections are
balanced across plain and Lombard conditions and are disjoint at the
`(speaker, utterance_code)` level. Query images are the frame displayed at the
temporal midpoint of the selected frontal clip. Opaque IDs are published; speaker,
condition, utterance code, transcription, and other text are not dataset features.

## Source audit and release discrepancies

The official JSON archive contains {source["metadata_rows"]:,} rows, while each
media archive contains {source["media_files_per_modality"]:,} files. All three
media archives have exactly the same recording filenames. The media restore 50
`s31` recordings absent from JSON. `s51` has only 40 plain and 50 Lombard files,
rather than the advertised 50 of each. `s14` has 51 plain and 49 Lombard rows,
including a duplicate `prwr3a` recording key. The JSON metadata labels 69
recordings as `WRONG`. Media filenames mark 70: the additional one is among the
50 media-only `s31` recordings. All 70, plus the one legacy-named duplicate, are
excluded from selection. Four frontal MOV files lack a `moov` atom and are corrupt:
`s32_l_pwip9p`, `s32_p_bwwj2n`, `s33_l_pwajza`, and `s33_p_sgwq2s`.

The deterministic selection required {len(fallbacks)} fallback(s). The selected
protocol remains exactly balanced after exclusions.

## Media processing and validation

The released MOV files are already H.264/yuv420p visual-only clips, so
{protocol["corpus"] - len(remuxed)} selected corpus videos are copied without
transcoding. {len(remuxed)} video(s) required a lossless video-stream remux to
remove embedded audio. Separate source WAV files are copied without transcoding.
Only the query midpoint frames are newly encoded, as RGB PNG files.

All {media["query_images_fully_decoded"]} images, all
{media["corpus_videos_fully_decoded_with_torchcodec"]} complete videos, and all
{media["corpus_audio_fully_decoded_with_torchcodec"]} complete audio files were
decoded during construction. Matching speaker, condition, utterance code, and
source filename stems establish audiovisual correspondence. Selected video/audio
duration differences were checked and paired decoding succeeded. The source
paper reports correlation-based audiovisual alignment before utterance extraction;
this construction did **not** independently verify signal-level synchronization.

Configs follow MTEB's retrieval layout:

- `queries/test`: `id`, `image`
- `corpus/test`: `id`, `video`, `audio`
- `qrels/test`: `query-id`, `corpus-id`, `score`

## Evaluation

The primary metric is nDCG@10. Other standard MTEB retrieval metrics are reported
as secondary results. This multi-positive speaker-retrieval protocol is newly
derived, so there is no source-paper retrieval score to reproduce.

## License and attribution

The source corpus is released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Please attribute the original authors and cite the paper below. The MTEB task
metadata and this card both use `cc-by-4.0`.

## Responsible use

This task evaluates biometric speaker identity using faces and voices. Such
representations can create privacy, surveillance, demographic-bias, and
misidentification risks. Results should be treated as research measurements, not
as evidence that a system is suitable for identity decisions. Users should
consider participant consent, applicable biometric-data law, subgroup behavior,
and downstream misuse before training or deploying related systems.

## Citation

```bibtex
@article{{alghamdi2018corpus,
  author = {{Alghamdi, Najwa and Maddock, Steve and Marxer, Ricard and Barker, Jon and Brown, Guy J.}},
  title = {{A corpus of audio-visual Lombard speech with frontal and profile views}},
  journal = {{The Journal of the Acoustical Society of America}},
  volume = {{143}},
  number = {{6}},
  pages = {{EL523--EL529}},
  year = {{2018}},
  doi = {{10.1121/1.5042758}},
}}
```
"""


def _prepare_source(
    work_dir: Path,
    source_dir: Path,
    *,
    download: bool,
    retries: int,
    workers: int,
) -> tuple[Path, list[Recording], dict[str, Any], dict[str, Any]]:
    archives = _ensure_sources(source_dir, download=download, retries=retries)
    extraction_root = work_dir / "extracted"
    archive_layout = _ensure_extracted(archives, extraction_root)
    metadata_rows = _load_metadata_rows(extraction_root)
    media_root = extraction_root / "lombardgrid"
    recordings, reconciliation = _source_reconciliation(media_root, metadata_rows)
    audit = _source_media_audit(
        media_root,
        work_dir / "reports",
        workers=workers,
    )
    source_summary = {
        "zenodo_record": _ZENODO_RECORD,
        "paper": _PAPER,
        "license": "cc-by-4.0",
        "archives": _archive_fingerprint(),
        "archive_layout": archive_layout,
        "source_reconciliation": reconciliation,
        "source_media_audit": audit,
    }
    reports_dir = work_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "source_summary.json").write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return media_root, recordings, reconciliation, audit


def _build_or_validate(args: argparse.Namespace, *, build: bool) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else work_dir / "source"
    output_dir = work_dir / "output"
    media_root, recordings, reconciliation, audit = _prepare_source(
        work_dir,
        source_dir,
        download=args.download,
        retries=args.download_retries,
        workers=args.workers,
    )
    unavailable = {Path(relative).stem for relative in audit["decode_errors"]}
    selected, fallbacks = _select_protocol(recordings, unavailable)
    qrels = _build_qrels(selected)
    protocol_validation = _validate_protocol(selected, qrels)

    if build:
        paths, remuxed = _materialize_media(
            media_root,
            output_dir,
            selected,
            workers=args.workers,
        )
    else:
        paths = _output_paths(output_dir, selected)
        remuxed = []
        missing = [
            str(path)
            for item_paths in paths.values()
            for path in item_paths.values()
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Validation requires a completed build; missing {len(missing)} files"
            )

    selected_media_validation = _validate_output_media(
        selected,
        paths,
        workers=args.workers,
    )
    corpus, queries, qrel_dataset = _make_datasets(selected, qrels, paths)
    expected_columns = {
        "queries": ["id", "image"],
        "corpus": ["id", "video", "audio"],
        "qrels": ["query-id", "corpus-id", "score"],
    }
    actual_columns = {
        "queries": queries.column_names,
        "corpus": corpus.column_names,
        "qrels": qrel_dataset.column_names,
    }
    if actual_columns != expected_columns:
        raise RuntimeError(f"Unexpected schemas: {actual_columns}")

    summary = {
        "archive_fingerprint": _archive_fingerprint(),
        "selection_seed": _SELECTION_SEED,
        "source_reconciliation": reconciliation,
        "source_media_audit": audit,
        "selection_fallbacks": fallbacks,
        "corpus_videos_remuxed_to_strip_audio": remuxed,
        "protocol_validation": protocol_validation,
        "selected_media_validation": selected_media_validation,
        "schemas": {
            "queries": _feature_schema(queries),
            "corpus": _feature_schema(corpus),
            "qrels": _feature_schema(qrel_dataset),
        },
        "dataset_examples": {
            "queries": {"id": queries[0]["id"], "image": "<image>"},
            "corpus": {
                "id": corpus[0]["id"],
                "video": "<video>",
                "audio": "<audio>",
            },
            "qrels": qrel_dataset[0],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps([asdict(item) for item in selected], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _dataset_card(summary),
        encoding="utf-8",
    )
    if build:
        _save_datasets(
            output_dir,
            corpus,
            queries,
            qrel_dataset,
            force=args.force,
        )
    else:
        for name, expected in (
            ("corpus", corpus),
            ("queries", queries),
            ("qrels", qrel_dataset),
        ):
            loaded = load_from_disk(str(output_dir / "datasets" / name))
            if not _dataset_identity_matches(name, loaded, expected):
                raise RuntimeError(f"Saved {name} dataset does not match rebuilt data")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _publish(work_dir: Path, repo_id: str) -> str:
    output_dir = work_dir / "output"
    required = [
        output_dir / "README.md",
        output_dir / "build_summary.json",
        output_dir / "datasets" / "queries",
        output_dir / "datasets" / "corpus",
        output_dir / "datasets" / "qrels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Build and validate before pushing; missing {missing}")
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face token found; run `hf auth login` first")

    corpus = load_from_disk(str(output_dir / "datasets" / "corpus"))
    queries = load_from_disk(str(output_dir / "datasets" / "queries"))
    qrels = load_from_disk(str(output_dir / "datasets" / "qrels"))
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    api = HfApi(token=token)
    # Upload the human-authored card first. Each Dataset push then merges its
    # generated dataset_info/configs into the card's YAML front matter. Uploading
    # the card last would discard those config declarations.
    api.upload_file(
        path_or_fileobj=(output_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add Lombard GRID dataset card",
    )
    DatasetDict({"test": queries}).push_to_hub(
        repo_id,
        "queries",
        token=token,
        commit_message="Add Lombard GRID image queries",
    )
    DatasetDict({"test": corpus}).push_to_hub(
        repo_id,
        "corpus",
        token=token,
        max_shard_size="500MB",
        commit_message="Add Lombard GRID video+audio corpus",
    )
    DatasetDict({"test": qrels}).push_to_hub(
        repo_id,
        "qrels",
        token=token,
        commit_message="Add Lombard GRID relevance judgments",
    )
    sha = api.dataset_info(repo_id).sha
    (output_dir / "hub_revision.txt").write_text(f"{sha}\n", encoding="utf-8")
    print(f"Published immutable revision {sha}", flush=True)
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("download", "inspect", "build", "validate", "push"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/lombard-grid-i2va"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing the four source ZIPs (default: WORK_DIR/source).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download any missing source archive for inspect/build/validate.",
    )
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-id", default="Cerru02/LombardGrid-I2VA")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else work_dir / "source"
    if args.action == "download":
        _ensure_sources(source_dir, download=True, retries=args.download_retries)
    elif args.action == "inspect":
        _, _, reconciliation, audit = _prepare_source(
            work_dir,
            source_dir,
            download=args.download,
            retries=args.download_retries,
            workers=args.workers,
        )
        print(
            json.dumps(
                {"source_reconciliation": reconciliation, "source_media_audit": audit},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.action == "build":
        _build_or_validate(args, build=True)
    elif args.action == "validate":
        _build_or_validate(args, build=False)
    elif args.action == "push":
        _publish(work_dir, args.repo_id)


if __name__ == "__main__":
    main()
