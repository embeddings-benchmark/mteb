#!/usr/bin/env python3
"""Build the FineGrainOCR image+text clustering evaluation subset.

The official ZIP is about 50 GB, but supports HTTP byte ranges. This builder
downloads its ZIP index, the contiguous validation OCR span, and only ranges
covering the selected validation images. Downloads are cached and CRC checked.

Without ``--push``, the processed DatasetDict is saved below ``--output-dir``.
With ``--push``, it is also uploaded to ``--repo-id``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import DatasetCard, HfApi, create_repo
from PIL import Image as PILImage
from PIL import ImageOps
from tqdm.auto import tqdm

from scripts.data.finegrainocr_clustering.analyze_archive import (
    ZipEntry,
    _extract_from_span,
    _group_members,
    _select_validation_keys,
    parse_central_directory,
)

SOURCE_URL = (
    "https://www.dropbox.com/scl/fi/jraqxgrg0z7carmj7anxs/"
    "FineGrainOCR.zip?rlkey=qq9p7orig0csxo7s1vq1htc5r&dl=1"
)
SOURCE_ARCHIVE_BYTES = 53_728_458_454
CENTRAL_DIRECTORY_START = 53_701_780_340
CENTRAL_DIRECTORY_END = 53_728_458_356
CENTRAL_DIRECTORY_SHA256 = (
    "242eaa1c31c37a47957269ce598e11b1414dbfe1d154c72977952e2314cbbb8a"
)
VALIDATION_TEXT_START = 197_817_161
VALIDATION_TEXT_END = 244_164_912
VALIDATION_TEXT_SHA256 = (
    "aef86966828d1e3daec104363007da1858b6f13656ef8e907bcc6e01b878b9ff"
)
SOURCE_COMMIT = "9ce19719123fd33a994b103b6e91c37a640ce92b"
CAP_PER_CLASS = 20
SELECTION_SEED = 42
EXPECTED_ROWS = 4_919
EXPECTED_CLASSES = 256
MAX_IMAGE_EDGE = 512
JPEG_QUALITY = 90
BARCODE_PATTERN = re.compile(r"(?<!\d)(?:\d[\s-]*){7,13}\d(?!\d)")

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class DownloadChunk:
    """One inclusive-exclusive archive byte range and its selected members."""

    start: int
    end: int
    entries: tuple[ZipEntry, ...]

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def filename(self) -> str:
        return f"{self.start}-{self.end - 1}.bin"


def redact_barcodes(text: str) -> str:
    """Remove OCR digit sequences that can reveal a GTIN class label."""

    return BARCODE_PATTERN.sub("[BARCODE]", text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "mteb-finegrainocr-builder/1.0"
        _THREAD_LOCAL.session = session
    return session


def _download_range(
    start: int,
    end: int,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = 6,
) -> Path:
    """Download one inclusive-exclusive byte range, resuming a partial file."""

    expected_size = end - start
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_size and (
            expected_sha256 is None or _sha256(destination) == expected_sha256
        ):
            return destination
        raise ValueError(f"Unexpected cached file: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded > expected_size:
            raise ValueError(f"Oversized partial download: {partial}")
        if downloaded == expected_size:
            partial.replace(destination)
            break
        request_start = start + downloaded
        try:
            with _session().get(
                SOURCE_URL,
                headers={"Range": f"bytes={request_start}-{end - 1}"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored byte range with status {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {request_start}-{end - 1}/"
                if not content_range.startswith(expected_prefix):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
                with partial.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if partial.stat().st_size == expected_size:
                partial.replace(destination)
                break
            raise RuntimeError(
                f"Short download: {partial.stat().st_size} != {expected_size}"
            )
        except (OSError, requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))

    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"Failed to complete {destination}")
    if expected_sha256 is not None and _sha256(destination) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {destination}")
    return destination


def _load_descriptions(
    grouped: dict[tuple[str, str, str], dict[str, ZipEntry]],
    text_span: bytes,
) -> dict[tuple[str, str, str], str]:
    descriptions: dict[tuple[str, str, str], str] = {}
    for key, pair in grouped.items():
        if key[0] != "validation":
            continue
        payload = json.loads(
            _extract_from_span(pair["text"], text_span, VALIDATION_TEXT_START).decode(
                "utf-8"
            )
        )
        description = payload[0].get("description", "") if payload else ""
        descriptions[key] = description
    return descriptions


def plan_image_chunks(
    selected_entries: list[ZipEntry],
    all_entries: list[ZipEntry],
    *,
    max_gap_bytes: int,
    max_chunk_bytes: int,
) -> list[DownloadChunk]:
    """Coalesce selected member ranges while bounding extra transfer and memory."""

    ordered = sorted(all_entries, key=lambda entry: entry.local_header_offset)
    next_offset = {
        entry.filename: ordered[index + 1].local_header_offset
        for index, entry in enumerate(ordered[:-1])
    }
    intervals = sorted(
        (
            entry.local_header_offset,
            next_offset.get(entry.filename, SOURCE_ARCHIVE_BYTES),
            entry,
        )
        for entry in selected_entries
    )
    if not intervals:
        return []

    chunks: list[DownloadChunk] = []
    start, end, first = intervals[0]
    members = [first]
    for member_start, member_end, entry in intervals[1:]:
        can_merge = (
            member_start - end <= max_gap_bytes
            and member_end - start <= max_chunk_bytes
        )
        if can_merge:
            end = member_end
            members.append(entry)
        else:
            chunks.append(DownloadChunk(start, end, tuple(members)))
            start, end, members = member_start, member_end, [entry]
    chunks.append(DownloadChunk(start, end, tuple(members)))
    return chunks


def _download_chunks(
    chunks: list[DownloadChunk], cache_dir: Path, workers: int
) -> dict[str, Path]:
    chunk_dir = cache_dir / "image_ranges"
    paths = {chunk.filename: chunk_dir / chunk.filename for chunk in chunks}
    pending = [chunk for chunk in chunks if not paths[chunk.filename].exists()]
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_range,
                    chunk.start,
                    chunk.end,
                    paths[chunk.filename],
                ): chunk
                for chunk in pending
            }
            with tqdm(
                total=sum(chunk.size for chunk in pending),
                unit="B",
                unit_scale=True,
                desc="Image ranges",
            ) as progress:
                for future in as_completed(futures):
                    chunk = futures[future]
                    future.result()
                    progress.update(chunk.size)
    for chunk in chunks:
        path = paths[chunk.filename]
        if path.stat().st_size != chunk.size:
            raise ValueError(f"Unexpected cached range size: {path}")
    return paths


def _resize_jpeg(payload: bytes) -> bytes:
    with PILImage.open(io.BytesIO(payload)) as source:
        source.verify()
    with PILImage.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), PILImage.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )
    return output.getvalue()


def _process_images(
    chunks: list[DownloadChunk],
    chunk_paths: dict[str, Path],
    processed_dir: Path,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for chunk in tqdm(chunks, desc="CRC check and resize"):
        span = chunk_paths[chunk.filename].read_bytes()
        for entry in chunk.entries:
            output_name = hashlib.sha256(entry.filename.encode()).hexdigest() + ".jpg"
            output_path = processed_dir / output_name
            if not output_path.exists():
                raw = _extract_from_span(entry, span, chunk.start)
                output_path.write_bytes(_resize_jpeg(raw))
            output_paths[entry.filename] = output_path
    return output_paths


def _build_dataset(
    selected_keys: list[tuple[str, str, str]],
    grouped: dict[tuple[str, str, str], dict[str, ZipEntry]],
    descriptions: dict[tuple[str, str, str], str],
    image_paths: dict[str, Path],
) -> tuple[DatasetDict, dict[str, Any]]:
    label_names = sorted({key[1] for key in selected_keys})
    label_index = {label: index for index, label in enumerate(label_names)}
    rows: dict[str, list[Any]] = {
        "image": [],
        "text": [],
        "label": [],
        "sample_id": [],
    }
    redacted_rows = 0
    redacted_sequences = 0
    image_bytes_total = 0
    widths: list[int] = []
    heights: list[int] = []
    text_lengths: list[int] = []
    text_counts: Counter[str] = Counter()
    text_classes: dict[str, set[str]] = defaultdict(set)
    for _split, class_id, stem in selected_keys:
        key = ("validation", class_id, stem)
        image_entry = grouped[key]["image"]
        raw_text = descriptions[key]
        text = redact_barcodes(raw_text)
        redacted_rows += text != raw_text
        redacted_sequences += len(BARCODE_PATTERN.findall(raw_text))
        image_path = image_paths[image_entry.filename]
        image_bytes = image_path.read_bytes()
        image_bytes_total += len(image_bytes)
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            widths.append(image.width)
            heights.append(image.height)
        rows["image"].append({"bytes": image_bytes, "path": image_path.name})
        rows["text"].append(text)
        rows["label"].append(label_index[class_id])
        rows["sample_id"].append(stem)
        text_lengths.append(len(text))
        text_counts[text] += 1
        text_classes[text].add(class_id)

    features = Features(
        {
            "image": Image(),
            "text": Value("string"),
            "label": ClassLabel(names=label_names),
            "sample_id": Value("string"),
        }
    )
    dataset = DatasetDict({"test": Dataset.from_dict(rows, features=features)})
    counts = Counter(rows["label"])
    ordered_text_lengths = sorted(text_lengths)
    p95_index = min(len(ordered_text_lengths) - 1, int(0.95 * len(text_lengths)))
    summary = {
        "source_commit": SOURCE_COMMIT,
        "selection_seed": SELECTION_SEED,
        "cap_per_class": CAP_PER_CLASS,
        "rows": len(dataset["test"]),
        "classes": len(label_names),
        "min_rows_per_class": min(counts.values()),
        "max_rows_per_class": max(counts.values()),
        "barcode_redacted_rows": redacted_rows,
        "barcode_redacted_sequences": redacted_sequences,
        "text_characters": {
            "min": min(text_lengths),
            "median": statistics.median(text_lengths),
            "p95": ordered_text_lengths[p95_index],
            "max": max(text_lengths),
        },
        "exact_duplicate_text_rows": sum(count - 1 for count in text_counts.values()),
        "cross_class_exact_duplicate_text_groups": sum(
            len(classes) > 1 for classes in text_classes.values()
        ),
        "image_max_edge": MAX_IMAGE_EDGE,
        "jpeg_quality": JPEG_QUALITY,
        "processed_image_bytes": image_bytes_total,
        "image_width": {
            "min": min(widths),
            "median": statistics.median(widths),
            "max": max(widths),
        },
        "image_height": {
            "min": min(heights),
            "median": statistics.median(heights),
            "max": max(heights),
        },
    }
    return dataset, summary


def _validate_dataset(dataset: DatasetDict) -> None:
    rows = dataset["test"]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(rows)}")
    if len(set(rows["label"])) != EXPECTED_CLASSES:
        raise ValueError("Unexpected class count")
    if any(not text.strip() for text in rows["text"]):
        raise ValueError("Empty OCR text survived selection")
    if any(BARCODE_PATTERN.search(text) for text in rows["text"]):
        raise ValueError("Barcode-like digit sequence survived redaction")
    for row in tqdm(rows, desc="Validate processed images"):
        image = row["image"]
        image.load()
        if max(image.size) > MAX_IMAGE_EDGE:
            raise ValueError(f"Oversized processed image: {image.size}")


def _save_local(dataset: DatasetDict, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    dataset.save_to_disk(output_dir)
    print(f"saved={output_dir}")


def _card_text(summary: dict[str, Any]) -> str:
    template = Path(__file__).with_name("DATASET_CARD.md").read_text()
    return template.replace("{{SUMMARY_JSON}}", json.dumps(summary, indent=2))


def _push(dataset: DatasetDict, repo_id: str, summary: dict[str, Any]) -> str:
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(
        repo_id,
        max_shard_size="500MB",
        commit_message="Add FineGrainOCR image-text clustering subset",
    )
    DatasetCard(_card_text(summary)).push_to_hub(
        repo_id,
        repo_type="dataset",
        commit_message="Document FineGrainOCR clustering subset",
    )
    revision = HfApi().dataset_info(repo_id).sha
    print(f"pushed={repo_id}@{revision}")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/finegrainocr-it-clustering"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/finegrainocr-it-clustering/dataset"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-gap-mib",
        type=int,
        default=4,
        help="Merge selected ranges separated by at most this many MiB.",
    )
    parser.add_argument(
        "--max-chunk-mib",
        type=int,
        default=64,
        help="Maximum coalesced range size and processing-memory bound.",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.push and not args.repo_id:
        parser.error("--repo-id is required with --push")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    central_directory = _download_range(
        CENTRAL_DIRECTORY_START,
        CENTRAL_DIRECTORY_END,
        args.cache_dir / "central-directory.bin",
        expected_sha256=CENTRAL_DIRECTORY_SHA256,
    )
    validation_text = _download_range(
        VALIDATION_TEXT_START,
        VALIDATION_TEXT_END,
        args.cache_dir / "validation-text.bin",
        expected_sha256=VALIDATION_TEXT_SHA256,
    )

    entries = parse_central_directory(central_directory)
    _, grouped = _group_members(entries)
    descriptions = _load_descriptions(grouped, validation_text.read_bytes())
    eligible = {key for key, text in descriptions.items() if text.strip()}
    selected_keys = _select_validation_keys(
        grouped,
        CAP_PER_CLASS,
        SELECTION_SEED,
        eligible_keys=eligible,
    )
    if len(selected_keys) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected selected row count: {len(selected_keys)}")
    selected_entries = [grouped[key]["image"] for key in selected_keys]
    chunks = plan_image_chunks(
        selected_entries,
        entries,
        max_gap_bytes=args.max_gap_mib * 1024 * 1024,
        max_chunk_bytes=args.max_chunk_mib * 1024 * 1024,
    )
    print(
        f"selected={len(selected_keys)} classes={len({key[1] for key in selected_keys})} "
        f"image_ranges={len(chunks)} transfer_bytes={sum(c.size for c in chunks)}"
    )
    chunk_paths = _download_chunks(chunks, args.cache_dir, args.workers)
    image_paths = _process_images(
        chunks,
        chunk_paths,
        args.cache_dir / "processed_images",
    )
    dataset, summary = _build_dataset(selected_keys, grouped, descriptions, image_paths)
    summary["selected_source_compressed_bytes"] = sum(
        grouped[key][modality].compressed_size
        for key in selected_keys
        for modality in ("image", "text")
    )
    summary["coalesced_image_transfer_bytes"] = sum(chunk.size for chunk in chunks)
    summary["image_archive_ranges"] = len(chunks)
    _validate_dataset(dataset)
    summary_path = args.cache_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _save_local(dataset, args.output_dir)
    if args.push:
        _push(dataset, args.repo_id, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
