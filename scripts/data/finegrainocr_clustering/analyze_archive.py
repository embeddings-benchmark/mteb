"""Inspect the FineGrainOCR ZIP index without downloading the 50 GB archive.

The Dropbox-hosted source archive supports HTTP byte ranges. Download its ZIP
central directory, then pass that file to this script to audit the split layout,
image/OCR pairing, class balance, and the transfer size of a class-balanced
subset before fetching any media.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import re
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
ZIP64_EXTRA_FIELD_ID = 0x0001
UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF

SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
MODALITY_NAMES = {
    "image": "image",
    "images": "image",
    "text": "text",
    "texts": "text",
}


@dataclass(frozen=True)
class ZipEntry:
    """Metadata required for selective extraction of one ZIP member."""

    filename: str
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    compression_method: int
    crc32: int


@dataclass(frozen=True)
class DatasetMember:
    """A FineGrainOCR image or OCR member inferred from its archive path."""

    split: str
    modality: str
    class_id: str
    stem: str
    entry: ZipEntry


def _zip64_values(extra: bytes) -> list[int]:
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise ValueError("Truncated ZIP extra-field header")
        field_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        data = extra[cursor : cursor + size]
        if len(data) != size:
            raise ValueError("Truncated ZIP extra-field payload")
        cursor += size
        if field_id == ZIP64_EXTRA_FIELD_ID:
            if len(data) % 8:
                raise ValueError("Unexpected ZIP64 extra-field size")
            return list(struct.unpack(f"<{len(data) // 8}Q", data))
    return []


def parse_central_directory(path: Path) -> list[ZipEntry]:
    """Parse a byte-for-byte ZIP central directory into member metadata."""

    data = path.read_bytes()
    entries: list[ZipEntry] = []
    cursor = 0
    fixed_header = struct.Struct("<4s6H3I5H2I")

    while cursor < len(data):
        if data[cursor : cursor + 4] != CENTRAL_DIRECTORY_SIGNATURE:
            raise ValueError(
                f"Unexpected signature at byte {cursor}; the input must contain "
                "only the ZIP central directory"
            )

        (
            _,
            _version_made_by,
            _version_needed,
            flags,
            compression_method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = fixed_header.unpack_from(data, cursor)

        variable_start = cursor + fixed_header.size
        filename_bytes = data[variable_start : variable_start + filename_length]
        extra_start = variable_start + filename_length
        extra = data[extra_start : extra_start + extra_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        filename = filename_bytes.decode(encoding)

        zip64 = iter(_zip64_values(extra))
        if uncompressed_size == UINT32_MAX:
            uncompressed_size = next(zip64)
        if compressed_size == UINT32_MAX:
            compressed_size = next(zip64)
        if local_header_offset == UINT32_MAX:
            local_header_offset = next(zip64)
        if disk_start == UINT16_MAX:
            next(zip64)

        entries.append(
            ZipEntry(
                filename=filename,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
                compression_method=compression_method,
                crc32=crc32,
            )
        )
        cursor = extra_start + extra_length + comment_length

    return entries


def infer_dataset_member(entry: ZipEntry) -> DatasetMember | None:
    """Infer split, modality, class, and sample stem from an archive path."""

    parts = PurePosixPath(entry.filename).parts
    folded = [part.casefold() for part in parts]

    split_index = next(
        (index for index, part in enumerate(folded) if part in SPLIT_NAMES),
        None,
    )
    modality_index = next(
        (index for index, part in enumerate(folded) if part in MODALITY_NAMES),
        None,
    )
    if split_index is None or modality_index is None:
        return None
    class_index = max(split_index, modality_index) + 1
    if class_index + 1 >= len(parts):
        return None

    suffix = PurePosixPath(parts[-1]).suffix.casefold()
    modality = MODALITY_NAMES[folded[modality_index]]
    expected_suffixes = {"image": {".jpg", ".jpeg", ".png"}, "text": {".json"}}
    if suffix not in expected_suffixes[modality]:
        return None

    return DatasetMember(
        split=SPLIT_NAMES[folded[split_index]],
        modality=modality,
        class_id=parts[class_index],
        stem=PurePosixPath(parts[-1]).stem,
        entry=entry,
    )


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0, "max": 0, "total": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "total": sum(values),
    }


def _group_members(
    entries: list[ZipEntry],
) -> tuple[
    list[DatasetMember],
    dict[tuple[str, str, str], dict[str, ZipEntry]],
]:
    members = [
        member
        for entry in entries
        if (member := infer_dataset_member(entry)) is not None
    ]
    grouped: dict[tuple[str, str, str], dict[str, ZipEntry]] = defaultdict(dict)
    for member in members:
        key = (member.split, member.class_id, member.stem)
        grouped[key][member.modality] = member.entry
    return members, grouped


def _select_validation_keys(
    grouped: dict[tuple[str, str, str], dict[str, ZipEntry]],
    cap_per_class: int,
    seed: int,
    eligible_keys: set[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    validation_by_class: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key, pair in grouped.items():
        split, class_id, stem = key
        if (
            split == "validation"
            and set(pair) == {"image", "text"}
            and (eligible_keys is None or key in eligible_keys)
        ):
            validation_by_class[class_id].append(key)

    selected: list[tuple[str, str, str]] = []
    for class_id in sorted(validation_by_class):
        ranked = sorted(
            validation_by_class[class_id],
            key=lambda key: hashlib.sha256(
                f"{seed}\0{class_id}\0{key[2]}".encode()
            ).digest(),
        )
        selected.extend(ranked[:cap_per_class])
    return selected


def validation_manifest(
    entries: list[ZipEntry], cap_per_class: int, seed: int
) -> list[dict[str, Any]]:
    """Create a deterministic manifest for selective range extraction."""

    _, grouped = _group_members(entries)
    selected_keys = _select_validation_keys(grouped, cap_per_class, seed)
    return [
        {
            "class_id": class_id,
            "stem": stem,
            "image": asdict(grouped[(split, class_id, stem)]["image"]),
            "text": asdict(grouped[(split, class_id, stem)]["text"]),
        }
        for split, class_id, stem in selected_keys
    ]


def _extract_from_span(entry: ZipEntry, span: bytes, span_offset: int) -> bytes:
    """Extract and CRC-check one member from a downloaded archive byte span."""

    relative_offset = entry.local_header_offset - span_offset
    if relative_offset < 0:
        raise ValueError(f"{entry.filename} starts before the supplied span")
    local_header = struct.Struct("<4s5H3I2H")
    (
        signature,
        _version,
        flags,
        compression_method,
        _modified_time,
        _modified_date,
        _crc32,
        _compressed_size,
        _uncompressed_size,
        filename_length,
        extra_length,
    ) = local_header.unpack_from(span, relative_offset)
    if signature != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"Invalid local-file signature for {entry.filename}")

    filename_start = relative_offset + local_header.size
    filename_bytes = span[filename_start : filename_start + filename_length]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    local_filename = filename_bytes.decode(encoding)
    if local_filename != entry.filename:
        raise ValueError(
            f"Central/local filename mismatch: {entry.filename!r} != {local_filename!r}"
        )

    compressed_start = filename_start + filename_length + extra_length
    compressed = span[compressed_start : compressed_start + entry.compressed_size]
    if len(compressed) != entry.compressed_size:
        raise ValueError(f"The supplied span truncates {entry.filename}")
    if compression_method == 0:
        raw = compressed
    elif compression_method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"Unsupported compression method {compression_method} for {entry.filename}"
        )
    if len(raw) != entry.uncompressed_size:
        raise ValueError(f"Uncompressed-size mismatch for {entry.filename}")
    if binascii.crc32(raw) != entry.crc32:
        raise ValueError(f"CRC mismatch for {entry.filename}")
    return raw


def _text_distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def audit_validation_text(
    entries: list[ZipEntry],
    span_path: Path,
    span_offset: int,
    cap_per_class: int,
    seed: int,
) -> dict[str, Any]:
    """Audit OCR quality and direct barcode-label leakage."""

    _, grouped = _group_members(entries)
    selected_keys = set(_select_validation_keys(grouped, cap_per_class, seed))
    span = span_path.read_bytes()
    locale_counts: Counter[str] = Counter()
    char_lengths: list[int] = []
    detection_counts: list[int] = []
    exact_descriptions: Counter[str] = Counter()
    description_classes: dict[str, set[str]] = defaultdict(set)
    empty_descriptions = 0
    label_digit_matches = 0
    barcode_like_texts = 0
    selected_empty_descriptions = 0
    selected_label_digit_matches = 0
    selected_char_lengths: list[int] = []
    nonempty_keys: set[tuple[str, str, str]] = set()
    descriptions: dict[tuple[str, str, str], str] = {}
    label_matches: dict[tuple[str, str, str], bool] = {}

    for key, pair in sorted(grouped.items()):
        split, class_id, _stem = key
        if split != "validation" or "text" not in pair:
            continue
        payload = json.loads(
            _extract_from_span(pair["text"], span, span_offset).decode("utf-8")
        )
        detection_counts.append(len(payload))
        first = payload[0] if payload else {}
        description = first.get("description", "")
        locale = first.get("locale") or "unspecified"
        locale_counts[locale] += 1
        char_lengths.append(len(description))
        exact_descriptions[description] += 1
        description_classes[description].add(class_id)
        descriptions[key] = description
        if description.strip():
            nonempty_keys.add(key)

        class_pattern = r"(?<!\d)" + r"[\s-]*".join(class_id) + r"(?!\d)"
        has_label_digits = bool(re.search(class_pattern, description))
        has_barcode_like_text = bool(
            re.search(r"(?<!\d)(?:\d[\s-]*){7,13}\d(?!\d)", description)
        )
        label_digit_matches += has_label_digits
        label_matches[key] = has_label_digits
        barcode_like_texts += has_barcode_like_text
        empty_descriptions += not description.strip()

        if key in selected_keys:
            selected_char_lengths.append(len(description))
            selected_empty_descriptions += not description.strip()
            selected_label_digit_matches += has_label_digits

    cross_class_duplicate_groups = sum(
        len(classes) > 1 for classes in description_classes.values()
    )
    duplicated_rows = sum(count - 1 for count in exact_descriptions.values())
    clean_selected_keys = _select_validation_keys(
        grouped,
        cap_per_class,
        seed,
        eligible_keys=nonempty_keys,
    )
    clean_class_counts = Counter(key[1] for key in clean_selected_keys)
    clean_source_bytes = sum(
        grouped[key][modality].compressed_size
        for key in clean_selected_keys
        for modality in ("image", "text")
    )
    return {
        "span": {
            "path": str(span_path),
            "offset": span_offset,
            "bytes": len(span),
        },
        "all_validation": {
            "samples": len(char_lengths),
            "empty_descriptions": empty_descriptions,
            "description_characters": _text_distribution(char_lengths),
            "detections_per_sample": _text_distribution(detection_counts),
            "first_detection_locales": dict(locale_counts.most_common()),
            "direct_class_id_digit_matches": label_digit_matches,
            "texts_with_barcode_like_digits": barcode_like_texts,
            "exact_duplicate_rows": duplicated_rows,
            "cross_class_exact_duplicate_groups": cross_class_duplicate_groups,
        },
        "selected_validation": {
            "samples": len(selected_keys),
            "empty_descriptions": selected_empty_descriptions,
            "direct_class_id_digit_matches": selected_label_digit_matches,
            "description_characters": _text_distribution(selected_char_lengths),
        },
        "recommended_nonempty_selection": {
            "samples": len(clean_selected_keys),
            "classes": len(clean_class_counts),
            "class_distribution": _distribution(list(clean_class_counts.values())),
            "source_compressed_bytes": clean_source_bytes,
            "direct_class_id_digit_matches_to_redact": sum(
                label_matches[key] for key in clean_selected_keys
            ),
            "description_characters": _text_distribution(
                [len(descriptions[key]) for key in clean_selected_keys]
            ),
        },
    }


def summarize(entries: list[ZipEntry], cap_per_class: int, seed: int) -> dict[str, Any]:
    members, grouped = _group_members(entries)
    class_counts: Counter[tuple[str, str]] = Counter()
    modalities: Counter[tuple[str, str]] = Counter()
    compressed_bytes: Counter[tuple[str, str]] = Counter()

    for member in members:
        class_counts[(member.split, member.class_id)] += member.modality == "image"
        modalities[(member.split, member.modality)] += 1
        compressed_bytes[(member.split, member.modality)] += (
            member.entry.compressed_size
        )

    paired_keys = sorted(
        key for key, pair in grouped.items() if set(pair) == {"image", "text"}
    )
    validation_by_class: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in paired_keys:
        split, class_id, _ = key
        if split == "validation":
            validation_by_class[class_id].append(key)
    selected_keys = _select_validation_keys(grouped, cap_per_class, seed)

    selected_compressed_bytes = sum(
        grouped[key][modality].compressed_size
        for key in selected_keys
        for modality in ("image", "text")
    )
    per_split_classes: dict[str, list[int]] = defaultdict(list)
    for (split, _class_id), count in class_counts.items():
        per_split_classes[split].append(count)

    paired_by_split = Counter(key[0] for key in paired_keys)
    all_group_keys = set(grouped)
    image_only = sum(set(pair) == {"image"} for pair in grouped.values())
    text_only = sum(set(pair) == {"text"} for pair in grouped.values())

    return {
        "archive": {
            "entries": len(entries),
            "compressed_bytes": sum(entry.compressed_size for entry in entries),
            "uncompressed_bytes": sum(entry.uncompressed_size for entry in entries),
        },
        "recognized_dataset_members": len(members),
        "modality_members": {
            f"{split}/{modality}": count
            for (split, modality), count in sorted(modalities.items())
        },
        "modality_compressed_bytes": {
            f"{split}/{modality}": count
            for (split, modality), count in sorted(compressed_bytes.items())
        },
        "paired_samples": dict(sorted(paired_by_split.items())),
        "pairing_anomalies": {
            "image_only": image_only,
            "text_only": text_only,
            "other": len(all_group_keys) - len(paired_keys) - image_only - text_only,
        },
        "image_class_distribution": {
            split: _distribution(counts)
            for split, counts in sorted(per_split_classes.items())
        },
        "balanced_validation_plan": {
            "cap_per_class": cap_per_class,
            "selection_seed": seed,
            "classes": len(validation_by_class),
            "paired_samples": len(selected_keys),
            "compressed_bytes": selected_compressed_bytes,
        },
        "first_entries": [asdict(entry) for entry in entries[:5]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "central_directory",
        type=Path,
        help="File containing only the ZIP central-directory byte range.",
    )
    parser.add_argument(
        "--cap-per-class",
        type=int,
        default=20,
        help="Maximum paired validation samples per class for the transfer plan.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed included in the stable SHA-256 ranking used for subsampling.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optionally write the selected validation member metadata as JSON.",
    )
    parser.add_argument(
        "--validation-text-span",
        type=Path,
        help="Optional contiguous archive span containing every validation OCR file.",
    )
    parser.add_argument(
        "--validation-text-span-offset",
        type=int,
        help="Absolute archive offset where --validation-text-span starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cap_per_class < 1:
        raise ValueError("--cap-per-class must be positive")
    entries = parse_central_directory(args.central_directory)
    summary = summarize(entries, args.cap_per_class, args.seed)
    if args.validation_text_span:
        text_members = [
            member
            for entry in entries
            if (member := infer_dataset_member(entry)) is not None
            and member.split == "validation"
            and member.modality == "text"
        ]
        inferred_offset = min(
            member.entry.local_header_offset for member in text_members
        )
        span_offset = (
            args.validation_text_span_offset
            if args.validation_text_span_offset is not None
            else inferred_offset
        )
        summary["validation_text_audit"] = audit_validation_text(
            entries,
            args.validation_text_span,
            span_offset,
            args.cap_per_class,
            args.seed,
        )
    print(json.dumps(summary, indent=2))
    if args.manifest_out:
        manifest = validation_manifest(entries, args.cap_per_class, args.seed)
        args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
