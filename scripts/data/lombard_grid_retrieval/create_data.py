#!/usr/bin/env python3
"""Build the Lombard GRID cross-modal utterance-retrieval datasets.

The script is reproducible, resumable, and non-publishing by default. It verifies
the official Zenodo sizes and MD5 digests, safely extracts the archives, audits
the source release, selects a balanced fixed protocol, materializes the selected
media, and validates every output with the media stack used by MTEB.

Examples:
  # Download and verify the four required source archives.
  python scripts/data/lombard_grid_retrieval/create_data.py download \
      --work-dir /tmp/lombard-grid-retrieval

  # Reconcile metadata and media, including a complete FFmpeg decode audit.
  python scripts/data/lombard_grid_retrieval/create_data.py inspect \
      --work-dir /tmp/lombard-grid-retrieval

  # Build and validate the shared media and link configs.
  python scripts/data/lombard_grid_retrieval/create_data.py build \
      --work-dir /tmp/lombard-grid-retrieval

  # Re-run validation without rebuilding media.
  python scripts/data/lombard_grid_retrieval/create_data.py validate \
      --work-dir /tmp/lombard-grid-retrieval

  # Publishing is a separate, explicit action.
  python scripts/data/lombard_grid_retrieval/create_data.py push \
      --work-dir /tmp/lombard-grid-retrieval \
      --repo-id Cerru02/LombardGrid-Retrieval
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
from typing import Any, cast

from datasets import Audio, Dataset, DatasetDict, Video, load_from_disk
from huggingface_hub import HfApi, create_repo, get_token

_ZENODO_RECORD = "https://zenodo.org/records/3736465"
_ZENODO_API = "https://zenodo.org/api/records/3736465/files"
_PAPER = "https://doi.org/10.1121/1.5042758"
_SELECTION_SEED = "mteb-lombard-grid-retrieval-v2"
_SPEAKERS = tuple(f"s{index}" for index in range(2, 56))
_CONDITIONS = ("p", "l")
_PAIRS_PER_SPEAKER = 10
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
class SelectedPair:
    speaker: str
    utterance_code: str
    plain_stem: str
    lombard_stem: str
    matching_condition: str


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
            return cast(dict[str, Any], cached)

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


def _stable_order_key(namespace: str, value: str) -> bytes:
    payload = f"{_SELECTION_SEED}\0{namespace}\0{value}".encode()
    return hashlib.sha256(payload).digest()


def _opaque_id(kind: str, stem: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{stem}".encode()).hexdigest()[:16]
    prefix = {
        "recording": "r",
        "audio": "a",
        "front_video": "vf",
        "profile_video": "vp",
    }[kind]
    return f"{prefix}_{digest}"


def _select_protocol(
    recordings: list[Recording],
    unavailable_stems: set[str],
) -> tuple[list[SelectedPair], list[dict[str, str]]]:
    candidates = [
        recording
        for recording in recordings
        if recording.status == "CORRECT" and not recording.legacy_filename
    ]
    by_key = {recording.canonical_key: recording for recording in candidates}
    if len(by_key) != len(candidates):
        raise RuntimeError("Duplicate eligible source recording keys")

    selected: list[SelectedPair] = []
    fallbacks: list[dict[str, str]] = []
    for speaker in _SPEAKERS:
        plain_codes = {
            code
            for spk, condition, code in by_key
            if spk == speaker and condition == "p"
        }
        lombard_codes = {
            code
            for spk, condition, code in by_key
            if spk == speaker and condition == "l"
        }
        ordered_codes = sorted(
            plain_codes & lombard_codes,
            key=lambda code: _stable_order_key("condition-pair", f"{speaker}_{code}"),
        )
        chosen: list[tuple[str, Recording, Recording]] = []
        for code in ordered_codes:
            plain = by_key[(speaker, "p", code)]
            lombard = by_key[(speaker, "l", code)]
            unavailable = sorted({plain.stem, lombard.stem} & unavailable_stems)
            if unavailable:
                fallbacks.append(
                    {
                        "speaker": speaker,
                        "utterance_code": code,
                        "rejected_stems": ",".join(unavailable),
                        "reason": "missing or corrupt required source modality",
                    }
                )
                continue
            chosen.append((code, plain, lombard))
            if len(chosen) == _PAIRS_PER_SPEAKER:
                break
        if len(chosen) != _PAIRS_PER_SPEAKER:
            raise RuntimeError(
                f"Unable to select {_PAIRS_PER_SPEAKER} complete plain/Lombard "
                f"pairs for {speaker}; found {len(chosen)}"
            )

        plain_matching_codes = {
            code
            for code, _, _ in sorted(
                chosen,
                key=lambda item: _stable_order_key(
                    "matching-condition", f"{speaker}_{item[0]}"
                ),
            )[: _PAIRS_PER_SPEAKER // 2]
        }
        selected.extend(
            SelectedPair(
                speaker=speaker,
                utterance_code=code,
                plain_stem=plain.stem,
                lombard_stem=lombard.stem,
                matching_condition="p" if code in plain_matching_codes else "l",
            )
            for code, plain, lombard in chosen
        )
    return selected, fallbacks


def _selected_stems(selected: list[SelectedPair]) -> list[str]:
    return [stem for pair in selected for stem in (pair.plain_stem, pair.lombard_stem)]


def _copy_resumable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    partial = destination.with_name(destination.name + ".part")
    with source.open("rb") as input_file, partial.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
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
    selected: list[SelectedPair],
) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for stem in _selected_stems(selected):
        paths[stem] = {
            "front_video": output_dir
            / "media"
            / "front"
            / f"{_opaque_id('front_video', stem)}.mov",
            "profile_video": output_dir
            / "media"
            / "profile"
            / f"{_opaque_id('profile_video', stem)}.mov",
            "audio": output_dir
            / "media"
            / "audio"
            / f"{_opaque_id('audio', stem)}.wav",
        }
    return paths


def _materialize_media(
    media_root: Path,
    output_dir: Path,
    selected: list[SelectedPair],
    *,
    workers: int,
) -> tuple[dict[str, dict[str, Path]], list[str]]:
    paths = _output_paths(output_dir, selected)
    remuxed: list[str] = []
    for stem in _selected_stems(selected):
        for source_view, output_view in (
            ("front", "front_video"),
            ("side", "profile_video"),
        ):
            if _materialize_video(
                media_root / source_view / f"{stem}.mov",
                paths[stem][output_view],
            ):
                remuxed.append(f"{output_view}:{_opaque_id(output_view, stem)}")
        _copy_resumable(
            media_root / "audio" / f"{stem}.wav",
            paths[stem]["audio"],
        )
    return paths, remuxed


def _validate_output_media(
    selected: list[SelectedPair],
    paths: dict[str, dict[str, Path]],
    *,
    workers: int,
) -> dict[str, Any]:
    from torchcodec.decoders import (  # type: ignore[attr-defined]
        AudioDecoder,
        VideoDecoder,
    )

    def validate_recording(stem: str) -> dict[str, Any]:
        recording_paths = paths[stem]
        video_results: dict[str, Any] = {}
        for view in ("front_video", "profile_video"):
            video_path = recording_paths[view]
            if _stream_types(video_path) != ["video"]:
                raise RuntimeError(f"Published video is not visual-only: {video_path}")
            video_decoder = VideoDecoder(video_path, dimension_order="NHWC")
            frames = video_decoder[:]
            if frames.ndim != 4 or frames.shape[0] == 0:
                raise RuntimeError(f"No decoded frames in {video_path}")
            video_results[view] = {
                "duration_seconds": video_decoder.metadata.duration_seconds,
                "frames": int(frames.shape[0]),
                "sha256": _sha256(video_path),
            }

        audio_path = recording_paths["audio"]
        audio_decoder = AudioDecoder(audio_path)
        samples = audio_decoder.get_all_samples().data
        if samples.ndim != 2 or samples.shape[-1] == 0:
            raise RuntimeError(f"No decoded samples in {audio_path}")
        return {
            "stem": stem,
            **video_results,
            "audio_duration_seconds": audio_decoder.metadata.duration_seconds,
            "audio_samples": int(samples.shape[-1]),
            "audio_sha256": _sha256(audio_path),
        }

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        results = list(pool.map(validate_recording, _selected_stems(selected)))

    front_hashes = [item["front_video"]["sha256"] for item in results]
    profile_hashes = [item["profile_video"]["sha256"] for item in results]
    audio_hashes = [item["audio_sha256"] for item in results]
    if len(set(front_hashes)) != len(front_hashes):
        raise RuntimeError("Byte-identical frontal videos found")
    if len(set(profile_hashes)) != len(profile_hashes):
        raise RuntimeError("Byte-identical profile videos found")
    if len(set(audio_hashes)) != len(audio_hashes):
        raise RuntimeError("Byte-identical audio files found")

    duration_differences = {
        view: _distribution(
            [
                abs(item[view]["duration_seconds"] - item["audio_duration_seconds"])
                for item in results
            ]
        )
        for view in ("front_video", "profile_video")
    }
    return {
        "recordings": len(results),
        "front_videos_fully_decoded_with_torchcodec": len(results),
        "profile_videos_fully_decoded_with_torchcodec": len(results),
        "audio_files_fully_decoded_with_torchcodec": len(results),
        "byte_identical_front_videos": 0,
        "byte_identical_profile_videos": 0,
        "byte_identical_audio_files": 0,
        "video_audio_duration_absolute_difference_seconds": duration_differences,
        "signal_level_synchronization_independently_verified": False,
    }


def _validate_protocol(
    selected: list[SelectedPair],
) -> dict[str, Any]:
    if len(selected) != len(_SPEAKERS) * _PAIRS_PER_SPEAKER:
        raise RuntimeError(f"Unexpected selected-pair count: {len(selected)}")
    stems = _selected_stems(selected)
    if len(stems) != 1_080 or len(stems) != len(set(stems)):
        raise RuntimeError("Expected 1,080 unique selected recordings")

    speaker_balance: dict[str, Any] = {}
    for speaker in _SPEAKERS:
        pairs = [pair for pair in selected if pair.speaker == speaker]
        if len(pairs) != _PAIRS_PER_SPEAKER:
            raise RuntimeError(f"Unbalanced speaker {speaker}")
        codes = {pair.utterance_code for pair in pairs}
        if len(codes) != _PAIRS_PER_SPEAKER:
            raise RuntimeError(f"Duplicate selected sentence codes for {speaker}")
        if Counter(pair.matching_condition for pair in pairs) != {"p": 5, "l": 5}:
            raise RuntimeError(f"Unbalanced matching-task conditions for {speaker}")
        for pair in pairs:
            plain = _parse_recording(pair.plain_stem)
            lombard = _parse_recording(pair.lombard_stem)
            if (
                plain.speaker != speaker
                or lombard.speaker != speaker
                or plain.condition != "p"
                or lombard.condition != "l"
                or plain.utterance_code != pair.utterance_code
                or lombard.utterance_code != pair.utterance_code
            ):
                raise RuntimeError(f"Invalid plain/Lombard pair for {speaker}")
        speaker_balance[speaker] = {
            "condition_pairs": len(pairs),
            "matching_plain": 5,
            "matching_lombard": 5,
            "distinct_sentence_codes": len(codes),
        }

    return {
        "speakers": len(_SPEAKERS),
        "condition_pairs": len(selected),
        "published_media_recordings": len(stems),
        "matching_utterance_recordings": len(selected),
        "speaker_condition_balance": speaker_balance,
        "published_feature_labels": [],
        "tasks": {
            "LombardGridA2VRetrieval": {
                "queries": 540,
                "corpus": 1_080,
                "qrels": 1_080,
                "positives_per_query": 2,
            },
            "LombardGridV2ARetrieval": {
                "queries": 1_080,
                "corpus": 540,
                "qrels": 1_080,
                "positives_per_query": 1,
            },
            "LombardGridV2VRetrieval": {
                "queries": 540,
                "corpus": 540,
                "qrels": 540,
                "positives_per_query": 1,
            },
            "LombardGridVA2VARetrieval": {
                "queries": 540,
                "corpus": 540,
                "qrels": 540,
                "positives_per_query": 1,
            },
        },
    }


def _make_datasets(
    selected: list[SelectedPair],
    paths: dict[str, dict[str, Path]],
) -> tuple[Dataset, Dataset, Dataset]:
    stems = _selected_stems(selected)
    media = Dataset.from_dict(
        {
            "recording_id": [_opaque_id("recording", stem) for stem in stems],
            "audio_id": [_opaque_id("audio", stem) for stem in stems],
            "front_video_id": [_opaque_id("front_video", stem) for stem in stems],
            "profile_video_id": [_opaque_id("profile_video", stem) for stem in stems],
            "front_video": [str(paths[stem]["front_video"]) for stem in stems],
            "profile_video": [str(paths[stem]["profile_video"]) for stem in stems],
            "audio": [str(paths[stem]["audio"]) for stem in stems],
        }
    ).cast_column("front_video", Video())
    media = media.cast_column("profile_video", Video())
    media = media.cast_column("audio", Audio())

    matching_stems = [
        pair.plain_stem if pair.matching_condition == "p" else pair.lombard_stem
        for pair in selected
    ]
    matching = Dataset.from_dict(
        {
            "audio_id": [_opaque_id("audio", stem) for stem in matching_stems],
            "front_video_id": [
                _opaque_id("front_video", stem) for stem in matching_stems
            ],
            "profile_video_id": [
                _opaque_id("profile_video", stem) for stem in matching_stems
            ],
        }
    )
    condition_pairs = Dataset.from_dict(
        {
            "plain_recording_id": [
                _opaque_id("recording", pair.plain_stem) for pair in selected
            ],
            "lombard_recording_id": [
                _opaque_id("recording", pair.lombard_stem) for pair in selected
            ],
        }
    )
    return media, matching, condition_pairs


def _feature_schema(dataset: Dataset) -> dict[str, str]:
    return {name: repr(feature) for name, feature in dataset.features.items()}


def _dataset_identity_matches(
    name: str,
    existing: Dataset,
    expected: Dataset,
) -> bool:
    if len(existing) != len(expected) or existing.features != expected.features:
        return False
    identity_columns = tuple(
        column
        for column in expected.column_names
        if column == "recording_id" or column.endswith("_id")
    )
    return all(
        list(existing[column]) == list(expected[column]) for column in identity_columns
    )


def _save_datasets(
    output_dir: Path,
    datasets: dict[str, Dataset],
    *,
    force: bool,
) -> None:
    datasets_dir = output_dir / "datasets"
    for name, dataset in datasets.items():
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
    remuxed = summary["videos_remuxed_to_strip_audio"]
    return f"""---
license: cc-by-4.0
pretty_name: Lombard GRID Cross-Modal Utterance Retrieval
task_categories:
- any-to-any
tags:
- mteb
- moeb
- audio-video-retrieval
- cross-view-retrieval
- speech-retrieval
---

# Lombard GRID Cross-Modal Utterance Retrieval

Four MTEB/MOEB utterance-retrieval tasks derived from the
[Lombard GRID corpus]({_ZENODO_RECORD}): audio-to-video, video-to-audio,
frontal-to-profile video, and plain-to-Lombard video+audio. Relevance always
requires the same utterance recording or the same speaker-and-sentence pair;
speaker identity alone is never relevant.

The source paper introduced the corpus but did not define these retrieval tasks.
Relevance is derived from native recording, speaker, sentence-code, condition,
and camera-view labels.

## Frozen evaluation protocol

- Speakers: {protocol["speakers"]} (`s2` through `s55`; `s1` was excluded by the source)
- Selected sentence codes: {protocol["condition_pairs"]} (10 per speaker)
- Published media recordings: {protocol["published_media_recordings"]}
- Matching-task recordings: {protocol["matching_utterance_recordings"]} (5 plain and 5 Lombard per speaker)
- Selection seed: `{_SELECTION_SEED}`

For every speaker, eligible sentence codes present in both conditions and all
three media streams are ordered by SHA-256 over the fixed seed, speaker, and
sentence code. The first ten valid pairs are retained. Five plain and five
Lombard recordings per speaker are then selected deterministically for the three
same-recording tasks. Opaque IDs are published; speaker, condition, sentence
code, transcription, and other text are not encodable dataset features.

| Task | Queries | Corpus | Qrels | Positives/query |
|---|---:|---:|---:|---:|
| Audio to frontal/profile video | 540 | 1,080 | 1,080 | 2 |
| Frontal/profile video to audio | 1,080 | 540 | 1,080 | 1 |
| Frontal video to profile video | 540 | 540 | 540 | 1 |
| Plain frontal video+audio to Lombard frontal video+audio | 540 | 540 | 540 | 1 |

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

The deterministic selection required {len(fallbacks)} fallback(s); the selected
protocol remains exactly balanced after exclusions.

## Media processing and validation

The released MOV files are already H.264/yuv420p visual-only clips, so
{2 * protocol["published_media_recordings"] - len(remuxed):,} selected videos are copied without
transcoding. {len(remuxed)} video(s) required a lossless video-stream remux to
remove embedded audio. Separate source WAV files are copied without transcoding.

All {media["front_videos_fully_decoded_with_torchcodec"]} frontal videos,
{media["profile_videos_fully_decoded_with_torchcodec"]} profile videos, and
{media["audio_files_fully_decoded_with_torchcodec"]} audio files were decoded
completely with TorchCodec during construction. Matching source filename stems
establish audiovisual correspondence. Selected video/audio duration differences
were checked and paired decoding succeeded. The source paper reports
correlation-based audiovisual alignment before utterance extraction; this
construction did **not** independently verify signal-level synchronization.

The shared configs avoid storing the same media repeatedly across four tasks:

- `media/test`: opaque recording and modality IDs, frontal video, profile video, audio
- `matching_utterances/test`: opaque audio, frontal-video, and profile-video IDs
- `condition_pairs/test`: opaque plain- and Lombard-recording IDs

The MTEB task loaders project these configs into the standard query, corpus, and
qrel structures without exposing source labels to embedding models.

## Evaluation

The primary metric for all four tasks is nDCG@10. Other standard MTEB retrieval
metrics are reported as secondary results. These protocols are newly derived, so
there is no source-paper retrieval score to reproduce.

## License and attribution

The source corpus is released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Please attribute the original authors and cite the paper below. The MTEB task
metadata and this card both use `cc-by-4.0`.

## Responsible use

The source contains identifiable faces and voices. Although these tasks evaluate
utterance correspondence rather than biometric identification, the representations
may still encode identity and demographic attributes. Users should consider
participant consent, privacy, applicable biometric-data law, subgroup behavior,
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
    protocol_validation = _validate_protocol(selected)

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
    media, matching, condition_pairs = _make_datasets(selected, paths)
    datasets = {
        "media": media,
        "matching_utterances": matching,
        "condition_pairs": condition_pairs,
    }
    expected_columns = {
        "media": [
            "recording_id",
            "audio_id",
            "front_video_id",
            "profile_video_id",
            "front_video",
            "profile_video",
            "audio",
        ],
        "matching_utterances": [
            "audio_id",
            "front_video_id",
            "profile_video_id",
        ],
        "condition_pairs": ["plain_recording_id", "lombard_recording_id"],
    }
    actual_columns = {name: dataset.column_names for name, dataset in datasets.items()}
    if actual_columns != expected_columns:
        raise RuntimeError(f"Unexpected schemas: {actual_columns}")

    summary = {
        "archive_fingerprint": _archive_fingerprint(),
        "selection_seed": _SELECTION_SEED,
        "source_reconciliation": reconciliation,
        "source_media_audit": audit,
        "selection_fallbacks": fallbacks,
        "videos_remuxed_to_strip_audio": remuxed,
        "protocol_validation": protocol_validation,
        "selected_media_validation": selected_media_validation,
        "schemas": {
            name: _feature_schema(dataset) for name, dataset in datasets.items()
        },
        "dataset_examples": {
            "media": {
                "recording_id": media["recording_id"][0],
                "audio_id": media["audio_id"][0],
                "front_video_id": media["front_video_id"][0],
                "profile_video_id": media["profile_video_id"][0],
                "front_video": "<video>",
                "profile_video": "<video>",
                "audio": "<audio>",
            },
            "matching_utterances": matching[0],
            "condition_pairs": condition_pairs[0],
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
            datasets,
            force=args.force,
        )
    else:
        for name, expected in datasets.items():
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
        output_dir / "datasets" / "media",
        output_dir / "datasets" / "matching_utterances",
        output_dir / "datasets" / "condition_pairs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Build and validate before pushing; missing {missing}")
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face token found; run `hf auth login` first")

    media = load_from_disk(str(output_dir / "datasets" / "media"))
    matching = load_from_disk(str(output_dir / "datasets" / "matching_utterances"))
    condition_pairs = load_from_disk(str(output_dir / "datasets" / "condition_pairs"))
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
    DatasetDict({"test": media}).push_to_hub(
        repo_id,
        "media",
        token=token,
        max_shard_size="500MB",
        commit_message="Add Lombard GRID selected media",
    )
    DatasetDict({"test": matching}).push_to_hub(
        repo_id,
        "matching_utterances",
        token=token,
        commit_message="Add Lombard GRID matching-utterance links",
    )
    DatasetDict({"test": condition_pairs}).push_to_hub(
        repo_id,
        "condition_pairs",
        token=token,
        commit_message="Add Lombard GRID plain-Lombard links",
    )
    api.upload_file(
        path_or_fileobj=(output_dir / "build_summary.json"),
        path_in_repo="build_summary.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add Lombard GRID build summary",
    )
    sha = api.dataset_info(repo_id).sha
    if sha is None:
        raise RuntimeError("Hugging Face did not return a dataset revision")
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
        default=Path("/tmp/lombard-grid-retrieval"),
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
    parser.add_argument("--repo-id", default="Cerru02/LombardGrid-Retrieval")
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
