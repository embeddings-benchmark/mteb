#!/usr/bin/env python3
"""Build the XModBench-Lite multiple-choice retrieval datasets for MTEB.

The published Lite split contains six canonical configurations where Vision is
the union of Image and Video. MTEB assigns modalities at task level, so this
script emits ten concrete directions while retaining the canonical source
configuration in stable query IDs and metadata.

The default invocation downloads only the pinned JSONL metadata (~5 MB),
validates it, and prints construction statistics:

    uv run --no-sync python scripts/data/xmodbench/create_data.py

Packaging media for a local export or Hub upload requires the 30.8 GB source
archive plus extracted files:

    uv run --no-sync python scripts/data/xmodbench/create_data.py \
        --work-dir /path/with/at/least/70GB/free \
        --download-media --save-to-disk

    uv run --no-sync python scripts/data/xmodbench/create_data.py \
        --work-dir /path/with/at/least/70GB/free \
        --download-media --push --repo-id USER/XModBench-MTEB
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Image, Video
from huggingface_hub import DatasetCard, HfApi, hf_hub_download, snapshot_download

SOURCE_DATASET = "RyanWW/XModBench"
SOURCE_REVISION = "a679188cf062b9810d2e09c2edabc0b1aef9f244"
SOURCE_CODE = "XingruiWang/XModBench"
SOURCE_CODE_REVISION = "687a9230f3621f96eeca21f8c873418c57411022"
LITE_BUILDER = "XingruiWang/lmms-eval"
LITE_BUILDER_REVISION = "313bf8034184ad245fa10939080692e377dac0e3"

LETTERS = ("A", "B", "C", "D")
LITE_CONFIGS = ("a2t", "a2v", "t2a", "t2v", "v2a", "v2t")
EXPECTED_ROWS_PER_CONFIG = 1_000
EXPECTED_ROWS_PER_FAMILY = 200
EXPECTED_RETAINED_ROWS = 5_981

# These MP4s in the pinned official XModBench archive are truncated: their
# ``mdat`` payload ends before the matching audio and they have no ``moov``
# atom, so standard decoders cannot open them. Drop an entire question if it
# uses one of these files as either its condition or one of its four options.
# The explicit row manifest makes the deviation from XModBench-Lite auditable
# and ensures a changed source revision cannot silently alter the exclusions.
EXCLUDED_SOURCE_ROWS = {
    ("a2v", 104): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 170): ("Data/ExtremCountAV/a7cRojOdljw.mp4",),
    ("a2v", 309): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 316): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 324): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 516): ("Data/ExtremCountAV/uby2dcP6cmw.mp4",),
    ("a2v", 538): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 711): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("a2v", 921): ("Data/ExtremCountAV/sFnX5gB99r8.mp4",),
    ("a2v", 999): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("t2v", 111): (
        "Data/urbansas_samples_videos_filtered/rivera0923_00_9_2.95_10.00.mp4",
    ),
    ("t2v", 529): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("t2v", 545): ("Data/ExtremCountAV/a7cRojOdljw.mp4",),
    ("t2v", 700): ("Data/ExtremCountAV/hPuylJBmk_8.mp4",),
    ("v2a", 110): ("Data/ExtremCountAV/uby2dcP6cmw.mp4",),
    ("v2t", 228): ("Data/ExtremCountAV/sFnX5gB99r8.mp4",),
    ("v2t", 515): ("Data/ExtremCountAV/a7cRojOdljw.mp4",),
    ("v2t", 695): ("Data/ExtremCountAV/sFnX5gB99r8.mp4",),
    ("v2t", 810): ("Data/ExtremCountAV/sFnX5gB99r8.mp4",),
}
EXCLUDED_VIDEO_PATHS = frozenset(
    path for paths in EXCLUDED_SOURCE_ROWS.values() for path in paths
)
_TRUNCATED_VIDEO_REASON = (
    "The pinned official MP4 is truncated, lacks a moov atom, and is not "
    "decodable by FFmpeg."
)
EXCLUDED_VIDEO_REASONS = {
    path: _TRUNCATED_VIDEO_REASON
    for path in EXCLUDED_VIDEO_PATHS
    if path.startswith("Data/ExtremCountAV/")
}
EXCLUDED_VIDEO_REASONS[
    "Data/urbansas_samples_videos_filtered/rivera0923_00_9_2.95_10.00.mp4"
] = (
    "The pinned official MP4 can be inspected by FFmpeg, but TorchCodec 0.14 "
    "cannot seek to its first presentation timestamp, so MTEB decoding fails."
)
EXPECTED_DIRECTION_COUNTS = {
    "at2t": 1_000,
    "at2i": 617,
    "at2v": 373,
    "t2a": 1_000,
    "t2i": 617,
    "t2v": 379,
    "it2a": 617,
    "vt2a": 382,
    "it2t": 617,
    "vt2t": 379,
}

FAMILY_NAMES = {
    "01_perception": "perception",
    "02_spatial": "spatial",
    "03_speech": "linguistic",
    "04_temporal": "temporal",
    "05_exteral": "knowledge",
}

# Query-side media conditions are always accompanied by the semantic question
# text. Text conditions and questions collapse into one text query modality.
DIRECTION_CODES = {
    ("Audio", "Text"): "at2t",
    ("Audio", "Image"): "at2i",
    ("Audio", "Video"): "at2v",
    ("Text", "Audio"): "t2a",
    ("Text", "Image"): "t2i",
    ("Text", "Video"): "t2v",
    ("Image", "Audio"): "it2a",
    ("Video", "Audio"): "vt2a",
    ("Image", "Text"): "it2t",
    ("Video", "Text"): "vt2t",
}

_ANSWER_FORMAT_RE = re.compile(
    r"\s*(?:"
    r"Choose A, B, C, or D\."
    r"|Answer with A, B, C, or D\."
    r"|Answer with A, B, C, or D"
    r"|Answer the question with A, B, C, or D\."
    r"|Answer the question with A, B, C, or D"
    r")\s*$",
    flags=re.IGNORECASE,
)


@dataclass
class RetrievalParts:
    """Rows for the four standard MTEB retrieval dataset configurations."""

    queries: list[dict[str, Any]] = field(default_factory=list)
    corpus: list[dict[str, Any]] = field(default_factory=list)
    qrels: list[dict[str, Any]] = field(default_factory=list)
    top_ranked: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    """Converted retrieval directions and source metadata."""

    directions: dict[str, RetrievalParts]
    metadata: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]


def clean_question(question: str) -> str:
    """Remove only the answer-letter boilerplate found in the pinned Lite data."""
    cleaned = _ANSWER_FORMAT_RE.sub("", question).rstrip()
    if not cleaned:
        raise ValueError("Question became empty after removing answer boilerplate")
    return cleaned


def _canonical_config(condition_modality: str, candidate_modality: str) -> str:
    def collapse(modality: str) -> str:
        return {
            "Audio": "a",
            "Image": "v",
            "Video": "v",
            "Text": "t",
        }[modality]

    return f"{collapse(condition_modality)}2{collapse(candidate_modality)}"


def _family_and_subtask(subtask_path: str) -> tuple[str, str]:
    try:
        prefix, subtask = subtask_path.split("/", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Invalid subtask path: {subtask_path!r}") from error
    family = FAMILY_NAMES.get(prefix.casefold())
    if family is None:
        raise ValueError(f"Unknown XModBench family prefix: {prefix!r}")
    return family, subtask


def _media_value(
    value: str,
    modality: str,
    *,
    media_root: Path | None,
) -> str:
    if modality == "Text":
        if not value.strip():
            raise ValueError("Encountered an empty text input")
        return value

    relative_path = PurePosixPath(value)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or relative_path.parts[0] != "Data"
        or ".." in relative_path.parts
    ):
        raise ValueError(f"Unsafe or unexpected media path: {value!r}")
    if media_root is None:
        return value

    resolved = media_root.joinpath(*relative_path.parts).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing XModBench media file: {resolved}")
    return str(resolved)


def _validate_row(row: dict[str, Any], source_config: str, row_number: int) -> None:
    required = {
        "index",
        "subtask",
        "question",
        "conditions",
        "options",
        "correct_answer",
        "category",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(
            f"{source_config} row {row_number} is missing fields: {sorted(missing)}"
        )
    if row["index"] != row_number:
        raise ValueError(f"{source_config} row {row_number} has index {row['index']!r}")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{source_config} row {row_number} has no question")
    if set(row["options"]) != set(LETTERS):
        raise ValueError(
            f"{source_config} row {row_number} does not have exactly A-D options"
        )
    if row["correct_answer"] not in LETTERS:
        raise ValueError(
            f"{source_config} row {row_number} has invalid answer "
            f"{row['correct_answer']!r}"
        )

    condition = row["conditions"]
    options = row["options"]
    if set(condition) != {"modality", "input"}:
        raise ValueError(
            f"{source_config} row {row_number} has invalid condition fields"
        )
    option_modalities = {option.get("modality") for option in options.values()}
    if len(option_modalities) != 1:
        raise ValueError(f"{source_config} row {row_number} mixes candidate modalities")
    if any(set(option) != {"modality", "input"} for option in options.values()):
        raise ValueError(f"{source_config} row {row_number} has invalid option fields")

    candidate_modality = next(iter(option_modalities))
    modality_pair = (condition["modality"], candidate_modality)
    if modality_pair not in DIRECTION_CODES:
        raise ValueError(
            f"{source_config} row {row_number} has unsupported modalities "
            f"{modality_pair!r}"
        )
    actual_config = _canonical_config(*modality_pair)
    if actual_config != source_config:
        raise ValueError(f"{source_config} row {row_number} maps to {actual_config}")
    _family_and_subtask(row["subtask"])


def _exclusion_for_source_row(
    row: dict[str, Any], source_config: str, row_number: int
) -> dict[str, Any] | None:
    uses = []
    if row["conditions"]["input"] in EXCLUDED_VIDEO_PATHS:
        uses.append(("condition", row["conditions"]["input"]))
    uses.extend(
        (f"option:{letter}", option["input"])
        for letter, option in row["options"].items()
        if option["input"] in EXCLUDED_VIDEO_PATHS
    )

    key = (source_config, row_number)
    expected_paths = EXCLUDED_SOURCE_ROWS.get(key)
    actual_paths = tuple(path for _, path in uses)
    if expected_paths is None:
        if uses:
            raise ValueError(
                f"Undeclared malformed media reference in {source_config} row "
                f"{row_number}: {actual_paths}"
            )
        return None
    if actual_paths != expected_paths:
        raise ValueError(
            f"Exclusion manifest mismatch in {source_config} row {row_number}: "
            f"expected {expected_paths}, found {actual_paths}"
        )

    condition_modality = row["conditions"]["modality"]
    candidate_modality = row["options"][LETTERS[0]]["modality"]
    family, subtask = _family_and_subtask(row["subtask"])
    roles = [role for role, _ in uses]
    reasons = [EXCLUDED_VIDEO_REASONS[path] for path in actual_paths]
    return {
        "query_id": f"xmodbench_lite_{source_config}_{row_number:04d}",
        "source_config": source_config,
        "source_index": row_number,
        "direction": DIRECTION_CODES[(condition_modality, candidate_modality)],
        "family": family,
        "subtask": subtask,
        "source_subtask": row["subtask"],
        "invalid_media_paths": list(actual_paths),
        "invalid_media_uses": roles,
        "invalid_media_reasons": reasons,
        "affects_query_or_correct_answer": (
            "condition" in roles or f"option:{row['correct_answer']}" in roles
        ),
        "reason": " ".join(dict.fromkeys(reasons)),
    }


def convert_source_row(
    row: dict[str, Any],
    source_config: str,
    row_number: int,
    *,
    media_root: Path | None = None,
) -> tuple[
    str,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Convert one source row into MTEB query/corpus/qrel/top-ranked rows."""
    _validate_row(row, source_config, row_number)
    condition = row["conditions"]
    options = row["options"]
    condition_modality = condition["modality"]
    candidate_modality = options[LETTERS[0]]["modality"]
    direction = DIRECTION_CODES[(condition_modality, candidate_modality)]
    query_id = f"xmodbench_lite_{source_config}_{row_number:04d}"
    question = clean_question(row["question"])

    if condition_modality == "Text":
        condition_text = _media_value(
            condition["input"], condition_modality, media_root=media_root
        )
        query = {
            "id": query_id,
            "text": f"Context: {condition_text}\n\n{question}",
        }
    else:
        query = {
            "id": query_id,
            "text": question,
            condition_modality.casefold(): _media_value(
                condition["input"], condition_modality, media_root=media_root
            ),
        }

    corpus = []
    corpus_ids = []
    for letter in LETTERS:
        option = options[letter]
        corpus_id = f"{query_id}_{letter}"
        corpus_ids.append(corpus_id)
        corpus.append(
            {
                "id": corpus_id,
                candidate_modality.casefold(): _media_value(
                    option["input"], candidate_modality, media_root=media_root
                ),
            }
        )

    correct_id = f"{query_id}_{row['correct_answer']}"
    qrel = {"query-id": query_id, "corpus-id": correct_id, "score": 1}
    top_ranked = {"query-id": query_id, "corpus-ids": corpus_ids}
    family, subtask = _family_and_subtask(row["subtask"])
    metadata = {
        "query_id": query_id,
        "source_config": source_config,
        "source_index": row["index"],
        "direction": direction,
        "condition_modality": condition_modality,
        "candidate_modality": candidate_modality,
        "family": family,
        "subtask": subtask,
        "source_subtask": row["subtask"],
        "category": row["category"],
        "correct_answer": row["correct_answer"],
        "original_question": row["question"],
        "cleaned_question": question,
    }
    return direction, query, corpus, qrel, top_ranked, metadata


def build_from_source(
    source_dir: Path,
    *,
    media_root: Path | None = None,
) -> BuildResult:
    """Validate and convert all six pinned XModBench-Lite JSONL files."""
    directions = {code: RetrievalParts() for code in DIRECTION_CODES.values()}
    metadata = []
    exclusions = []
    family_config_counts: Counter[tuple[str, str]] = Counter()

    for source_config in LITE_CONFIGS:
        path = source_dir / "data_lite" / f"{source_config}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned Lite metadata: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != EXPECTED_ROWS_PER_CONFIG:
            raise ValueError(
                f"{source_config} has {len(rows)} rows; expected "
                f"{EXPECTED_ROWS_PER_CONFIG}"
            )

        for row_number, row in enumerate(rows):
            _validate_row(row, source_config, row_number)
            family, _ = _family_and_subtask(row["subtask"])
            family_config_counts[(family, source_config)] += 1
            exclusion = _exclusion_for_source_row(row, source_config, row_number)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue

            (
                direction,
                query,
                corpus,
                qrel,
                top_ranked,
                source_metadata,
            ) = convert_source_row(
                row,
                source_config,
                row_number,
                media_root=media_root,
            )
            parts = directions[direction]
            parts.queries.append(query)
            parts.corpus.extend(corpus)
            parts.qrels.append(qrel)
            parts.top_ranked.append(top_ranked)
            metadata.append(source_metadata)

    invalid_cells = {
        cell: count
        for cell, count in family_config_counts.items()
        if count != EXPECTED_ROWS_PER_FAMILY
    }
    if invalid_cells or len(family_config_counts) != 30:
        raise ValueError(
            f"Lite data is not balanced at 200 rows per family/config: {invalid_cells}"
        )
    _validate_build(directions, metadata, exclusions)
    return BuildResult(
        directions=directions,
        metadata=metadata,
        exclusions=exclusions,
    )


def _validate_build(
    directions: dict[str, RetrievalParts],
    metadata: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> None:
    if len(metadata) != EXPECTED_RETAINED_ROWS:
        raise ValueError(
            f"Built {len(metadata)} retained metadata rows instead of "
            f"{EXPECTED_RETAINED_ROWS}"
        )
    if len(exclusions) != len(EXCLUDED_SOURCE_ROWS):
        raise ValueError(
            f"Recorded {len(exclusions)} exclusions instead of "
            f"{len(EXCLUDED_SOURCE_ROWS)}"
        )
    if {(row["source_config"], row["source_index"]) for row in exclusions} != set(
        EXCLUDED_SOURCE_ROWS
    ):
        raise ValueError("Recorded exclusions do not match the exclusion manifest")

    actual_direction_counts = {
        direction: len(parts.queries) for direction, parts in directions.items()
    }
    if actual_direction_counts != EXPECTED_DIRECTION_COUNTS:
        raise ValueError(
            "Unexpected retained direction counts: "
            f"{actual_direction_counts} != {EXPECTED_DIRECTION_COUNTS}"
        )

    expected_config_counts = {
        "a2t": 1_000,
        "a2v": 990,
        "t2a": 1_000,
        "t2v": 996,
        "v2a": 999,
        "v2t": 996,
    }
    config_counts = Counter(row["source_config"] for row in metadata)
    if config_counts != expected_config_counts:
        raise ValueError(
            f"Unexpected retained canonical config counts: {config_counts}"
        )

    expected_family_counts = {
        "perception": 1_200,
        "spatial": 1_199,
        "linguistic": 1_200,
        "temporal": 1_182,
        "knowledge": 1_200,
    }
    family_counts = Counter(row["family"] for row in metadata)
    if family_counts != expected_family_counts:
        raise ValueError(f"Unexpected retained family counts: {family_counts}")

    all_query_ids = []
    for direction, parts in directions.items():
        query_ids = [row["id"] for row in parts.queries]
        corpus_ids = [row["id"] for row in parts.corpus]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Duplicate query IDs in {direction}")
        if len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError(f"Duplicate corpus IDs in {direction}")
        if len(parts.corpus) != 4 * len(parts.queries):
            raise ValueError(f"{direction} does not have four candidates per query")
        if not (len(parts.queries) == len(parts.qrels) == len(parts.top_ranked)):
            raise ValueError(f"Inconsistent retrieval row counts in {direction}")

        corpus_id_set = set(corpus_ids)
        for qrel, top_ranked in zip(parts.qrels, parts.top_ranked, strict=True):
            if qrel["corpus-id"] not in corpus_id_set:
                raise ValueError(f"Missing relevant document in {direction}")
            if len(top_ranked["corpus-ids"]) != 4:
                raise ValueError(f"Invalid top-ranked list in {direction}")
            if qrel["corpus-id"] not in top_ranked["corpus-ids"]:
                raise ValueError(f"Relevant document is not top-ranked in {direction}")
        all_query_ids.extend(query_ids)

    if len(all_query_ids) != len(set(all_query_ids)):
        raise ValueError("Query IDs collide across concrete directions")


def _cast_modality(dataset: Dataset, modality: str) -> Dataset:
    if modality == "audio":
        return dataset.cast_column(modality, Audio())
    if modality == "image":
        return dataset.cast_column(modality, Image())
    if modality == "video":
        return dataset.cast_column(modality, Video())
    return dataset


def _as_datasets(result: BuildResult) -> dict[str, DatasetDict]:
    output = {}
    for direction, parts in result.directions.items():
        query_modality, corpus_modality = direction.split("2", maxsplit=1)
        query_media = next(
            (
                modality
                for code, modality in (("a", "audio"), ("i", "image"), ("v", "video"))
                if code in query_modality
            ),
            None,
        )
        corpus_media = {
            "a": "audio",
            "i": "image",
            "v": "video",
            "t": None,
        }[corpus_modality]

        queries = Dataset.from_list(parts.queries)
        corpus = Dataset.from_list(parts.corpus)
        if query_media is not None:
            queries = _cast_modality(queries, query_media)
        if corpus_media is not None:
            corpus = _cast_modality(corpus, corpus_media)
        output[f"{direction}-queries"] = DatasetDict({"test": queries})
        output[f"{direction}-corpus"] = DatasetDict({"test": corpus})
        output[f"{direction}-qrels"] = DatasetDict(
            {"test": Dataset.from_list(parts.qrels)}
        )
        output[f"{direction}-top_ranked"] = DatasetDict(
            {"test": Dataset.from_list(parts.top_ranked)}
        )
    output["metadata"] = DatasetDict({"test": Dataset.from_list(result.metadata)})
    output["exclusions"] = DatasetDict({"test": Dataset.from_list(result.exclusions)})
    return output


def _print_summary(result: BuildResult) -> None:
    print(
        f"XModBench source: {SOURCE_DATASET}@{SOURCE_REVISION}\n"
        f"Source code: https://github.com/{SOURCE_CODE}/tree/"
        f"{SOURCE_CODE_REVISION}\n"
        f"Lite builder: https://github.com/{LITE_BUILDER}/tree/"
        f"{LITE_BUILDER_REVISION}"
    )
    print("\nConcrete retrieval directions:")
    for direction, parts in result.directions.items():
        print(
            f"  {direction:5s} queries={len(parts.queries):4d} "
            f"candidates={len(parts.corpus):4d}"
        )
    print(f"\nRetained queries: {len(result.metadata)}")
    print(f"Excluded source rows: {len(result.exclusions)}")
    print(
        "Retained canonical configs:",
        dict(Counter(m["source_config"] for m in result.metadata)),
    )
    print("Retained families:", dict(Counter(m["family"] for m in result.metadata)))


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = destination.joinpath(*PurePosixPath(member.filename).parts)
            if not member_path.resolve().is_relative_to(destination_root):
                raise ValueError(f"Unsafe path in source archive: {member.filename}")
        zip_file.extractall(destination)


def _download_source(
    work_dir: Path, *, download_media: bool
) -> tuple[Path, Path | None]:
    source_dir = Path(
        snapshot_download(
            repo_id=SOURCE_DATASET,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=["README.md", "data_lite/*.jsonl"],
            local_dir=work_dir / "source",
        )
    )
    if not download_media:
        return source_dir, None

    archive = Path(
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename="Data.zip",
            local_dir=source_dir,
        )
    )
    if not (source_dir / "Data").is_dir():
        print(f"Extracting {archive} into {source_dir}")
        _safe_extract(archive, source_dir)
    return source_dir, source_dir


def _save_datasets(datasets: dict[str, DatasetDict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for config_name, dataset in datasets.items():
        destination = output_dir / config_name
        dataset.save_to_disk(destination)
        print(f"Saved {config_name} to {destination}")


def _calculate_descriptive_stats(
    datasets: dict[str, DatasetDict],
    result: BuildResult,
    *,
    num_proc: int | None,
    directions: set[str] | None = None,
) -> None:
    """Calculate MTEB statistics from local media without redownloading Hub data."""
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.tasks.retrieval.eng.xmod_bench import (
        XModBenchAT2IRetrieval,
        XModBenchAT2TRetrieval,
        XModBenchAT2VRetrieval,
        XModBenchIT2ARetrieval,
        XModBenchIT2TRetrieval,
        XModBenchT2ARetrieval,
        XModBenchT2IRetrieval,
        XModBenchT2VRetrieval,
        XModBenchVT2ARetrieval,
        XModBenchVT2TRetrieval,
    )

    tasks = [
        XModBenchAT2TRetrieval(),
        XModBenchAT2IRetrieval(),
        XModBenchAT2VRetrieval(),
        XModBenchT2ARetrieval(),
        XModBenchT2IRetrieval(),
        XModBenchT2VRetrieval(),
        XModBenchIT2ARetrieval(),
        XModBenchVT2ARetrieval(),
        XModBenchIT2TRetrieval(),
        XModBenchVT2TRetrieval(),
    ]
    for task in tasks:
        direction = task.metadata.hf_subsets[0]
        if directions is not None and direction not in directions:
            continue
        parts = result.directions[direction]
        relevant_docs = {
            row["query-id"]: {row["corpus-id"]: row["score"]} for row in parts.qrels
        }
        top_ranked = {row["query-id"]: row["corpus-ids"] for row in parts.top_ranked}
        task.dataset = {
            direction: {
                "test": RetrievalSplitData(
                    queries=datasets[f"{direction}-queries"]["test"],
                    corpus=datasets[f"{direction}-corpus"]["test"],
                    relevant_docs=relevant_docs,
                    top_ranked=top_ranked,
                )
            }
        }
        task.data_loaded = True
        task.calculate_descriptive_statistics(overwrite_results=True, num_proc=num_proc)
        print(f"Calculated descriptive statistics for {task.metadata.name}")


def _card_body() -> str:
    return f"""
# XModBench-Lite for MTEB

This repository is a deterministic MTEB normalization of the official
[`RyanWW/XModBench`](https://huggingface.co/datasets/RyanWW/XModBench)
XModBench-Lite release at revision `{SOURCE_REVISION}`. The source contains
6,000 four-choice questions balanced across six canonical modality
configurations and five capability families.

This MTEB adaptation retains 5,981 questions. It excludes 19 questions that
reference five unusable MP4 files in the pinned official archive. Four are
truncated:
`a7cRojOdljw.mp4`, `hPuylJBmk_8.mp4`, `sFnX5gB99r8.mp4`, and
`uby2dcP6cmw.mp4`. The files lack a final MP4 index and their video payloads
end before the corresponding audio. The fifth,
`rivera0923_00_9_2.95_10.00.mp4`, cannot be sought to its first presentation
timestamp by TorchCodec 0.14. The `exclusions` configuration records every
omitted source row, media path, usage, and reason. Original source indices are
preserved in retained IDs.

Each question is represented as a retrieval problem with four candidates, one
relevant document, and a `top_ranked` list that restricts evaluation to the
original answer choices. Accuracy is therefore equivalent to the source
multiple-choice metric.

MTEB assigns modalities at task level, while XModBench uses Vision to mean the
union of Image and Video. This normalization consequently exposes ten concrete
directions: `at2t`, `at2i`, `at2v`, `t2a`, `t2i`, `t2v`, `it2a`, `vt2a`,
`it2t`, and `vt2t`. Each direction has `queries`, `corpus`, `qrels`, and
`top_ranked` configurations. The `metadata` configuration preserves source
indices, canonical configurations, families, subtasks, categories, modalities,
answers, and original questions; `exclusions` documents the 19 omitted rows.

Query-side media are accompanied by the semantic question text. For text
conditions, the query is formatted as `Context: {{condition}}` followed by the
source question, matching the authors' lmms-eval integration. The conversion
removes only the source's exact trailing A/B/C/D answer-format boilerplate.

## Reproducibility

Generated by `scripts/data/xmodbench/create_data.py` in MTEB from:

- dataset: `{SOURCE_DATASET}@{SOURCE_REVISION}`
- source code: [`{SOURCE_CODE}@{SOURCE_CODE_REVISION}`](https://github.com/{SOURCE_CODE}/tree/{SOURCE_CODE_REVISION})
- Lite builder: [`{LITE_BUILDER}@{LITE_BUILDER_REVISION}`](https://github.com/{LITE_BUILDER}/tree/{LITE_BUILDER_REVISION})

## License and citation

The source benchmark is released under the MIT License. Its authors note that
redistributed media remain subject to the licenses of their underlying source
datasets. Please review the [source dataset card](https://huggingface.co/datasets/RyanWW/XModBench)
before reuse.

```bibtex
@inproceedings{{wang2026xmodbench,
  title     = {{XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models}},
  author    = {{Wang, Xingrui and Liu, Jiang and Huang, Chao and Yu, Xiaodong and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yuille, Alan and Barsoum, Emad and Liu, Zicheng}},
  booktitle = {{International Conference on Learning Representations (ICLR)}},
  year      = {{2026}},
  url       = {{https://arxiv.org/abs/2510.15148}}
}}
```
"""


def _update_card(repo_id: str) -> None:
    card = DatasetCard.load(repo_id, repo_type="dataset")
    generated_metadata = card.content.split("# XModBench-Lite for MTEB", maxsplit=1)[0]
    card.content = generated_metadata.rstrip() + "\n" + _card_body()
    card.push_to_hub(
        repo_id,
        repo_type="dataset",
        commit_message="Document XModBench-Lite MTEB normalization",
    )


def _push_datasets(
    datasets: dict[str, DatasetDict],
    repo_id: str,
    *,
    config_names: set[str] | None = None,
) -> None:
    if config_names is not None:
        unknown = config_names - datasets.keys()
        if unknown:
            raise ValueError(f"Unknown dataset configurations: {sorted(unknown)}")
    for config_name, dataset in datasets.items():
        if config_names is not None and config_name not in config_names:
            continue
        dataset.push_to_hub(repo_id, config_name=config_name)
        print(f"Pushed {repo_id}/{config_name}")
    _update_card(repo_id)
    revision = HfApi().dataset_info(repo_id).sha
    print(f"Pushed {repo_id}@{revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/xmodbench-mteb"),
        help="Download, extraction, and local export directory",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing source directory containing data_lite/*.jsonl",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Existing source root containing Data/",
    )
    parser.add_argument(
        "--download-media",
        action="store_true",
        help="Download and extract the pinned 30.8 GB media archive",
    )
    parser.add_argument(
        "--save-to-disk",
        action="store_true",
        help="Save all processed Hugging Face configurations locally",
    )
    parser.add_argument(
        "--calculate-stats",
        action="store_true",
        help="Calculate MTEB descriptive statistics from the local media",
    )
    parser.add_argument(
        "--stats-workers",
        type=int,
        default=None,
        help="Worker threads used to hash media for --calculate-stats",
    )
    parser.add_argument(
        "--stats-directions",
        nargs="+",
        choices=sorted(DIRECTION_CODES.values()),
        help="Only calculate statistics for these concrete directions",
    )
    parser.add_argument("--push", action="store_true", help="Push configs to Hub")
    parser.add_argument("--repo-id", help="Destination Hub dataset ID for --push")
    parser.add_argument(
        "--push-configs",
        nargs="+",
        help="Only push these generated configurations",
    )
    args = parser.parse_args()

    if args.download_media and (args.source_dir or args.media_root):
        parser.error(
            "--download-media cannot be combined with --source-dir or --media-root"
        )
    if args.push and not args.repo_id:
        parser.error("--repo-id is required with --push")
    if args.push_configs and not args.push:
        parser.error("--push-configs requires --push")

    if args.source_dir is None:
        source_dir, downloaded_media_root = _download_source(
            args.work_dir, download_media=args.download_media
        )
    else:
        source_dir = args.source_dir
        downloaded_media_root = None
    media_root = args.media_root or downloaded_media_root
    if (args.save_to_disk or args.push or args.calculate_stats) and media_root is None:
        parser.error(
            "Media packaging and statistics require --download-media or an "
            "existing --media-root"
        )

    result = build_from_source(source_dir, media_root=media_root)
    _print_summary(result)
    if not args.save_to_disk and not args.push and not args.calculate_stats:
        return

    datasets = _as_datasets(result)
    if args.calculate_stats:
        _calculate_descriptive_stats(
            datasets,
            result,
            num_proc=args.stats_workers,
            directions=set(args.stats_directions) if args.stats_directions else None,
        )
    if args.save_to_disk:
        _save_datasets(datasets, args.work_dir / "processed")
    if args.push:
        assert args.repo_id is not None
        _push_datasets(
            datasets,
            args.repo_id,
            config_names=set(args.push_configs) if args.push_configs else None,
        )


if __name__ == "__main__":
    main()
