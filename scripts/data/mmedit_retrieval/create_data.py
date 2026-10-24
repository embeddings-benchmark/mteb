#!/usr/bin/env python3
"""Audit the public MMEdit-TestSet source data for an MTEB retrieval task.

The default mode is intentionally audit-only. Dataset construction and remote
upload require explicit flags.

Usage:
  uv run python scripts/data/mmedit_retrieval/create_data.py

  # Keep the Hugging Face cache at a specific location outside this repository.
  uv run python scripts/data/mmedit_retrieval/create_data.py \
      --cache-dir /path/to/huggingface-cache

  # Re-audit an already-downloaded snapshot without network access.
  uv run python scripts/data/mmedit_retrieval/create_data.py \
      --source-dir /path/to/MMEdit-TestSet --offline

  # Construct and inspect the MTEB dataset locally, without uploading.
  uv run python scripts/data/mmedit_retrieval/create_data.py \
      --source-dir /path/to/MMEdit-TestSet \
      --build --output-dir /tmp/mmedit-mteb

  # Upload only after local validation. HF_TOKEN is read from the environment.
  HF_TOKEN=... uv run python scripts/data/mmedit_retrieval/create_data.py \
      --source-dir /path/to/MMEdit-TestSet --build --push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from wave import Error as WaveError
from wave import open as wave_open

from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, create_repo, dataset_info, snapshot_download
from tqdm import tqdm

SOURCE_REPO_ID = "CocoBro/MMEdit-TestSet"
SOURCE_REVISION = "ae4f9a772180a2a3c77c2e865b398e7d6f60bcee"
DEFAULT_OUTPUT_REPO_ID = "pranitchawla/MMEdit-AT2A"
EXPECTED_TRIPLETS = 3_317
EXPECTED_QRELS = 3_337
EXPECTED_SAMPLE_RATE = 24_000
EXPECTED_CHANNELS = 1
EXPECTED_DURATION_SECONDS = 10.0
_DURATION_TOLERANCE_SECONDS = 0.05
_AUDIO_ID_RE = re.compile(r"^(?P<family>.+)_(?P<number>\d+)$")
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DATASET_CARD = """---
license: apache-2.0
language:
- en
task_categories:
- audio-to-audio
tags:
- mteb
- audio-retrieval
- composed-retrieval
configs:
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
---

# MMEdit-AT2A

MMEdit-AT2A is an MTEB-formatted composed audio retrieval adaptation of
[`CocoBro/MMEdit-TestSet`](https://huggingface.co/datasets/CocoBro/MMEdit-TestSet).
Each query combines an unedited source recording with a natural-language edit
instruction, and the corpus contains the corresponding edited recordings.

## Schema

- `queries`: `id`, `audio` (unedited source), and `text` (edit instruction)
- `corpus`: `id` and `audio` (edited target)
- `qrels`: `query-id`, `corpus-id`, and binary `score`

The test split contains 3,317 queries and 3,317 corpus items. Exact
byte-identical target duplicates are all marked relevant, producing 3,337 qrels.

## Construction

The source is pinned to commit
`ae4f9a772180a2a3c77c2e865b398e7d6f60bcee`. Triplets are joined by `audio_id`.
The construction preserves the original WAV payloads without padding,
resampling, or loudness normalization. IDs are deterministic:
`q-{audio_id}` for queries and `t-{audio_id}` for corpus items.

## Known source-format discrepancies

Although the source card describes all clips as 24 kHz, mono, and ten seconds:

- 1,394 source files and 1,394 target files differ from ten seconds by more
  than 0.05 seconds (minimum observed duration: 2.3985 seconds).
- 143 targets (`replace_one` and `replace_time`) are 16 kHz, 32-bit PCM rather
  than 24 kHz, 16-bit PCM.
- All 6,634 WAV files are mono.

These properties are retained so this derivative stays faithful to the source.

## Benchmark limitation: source-only leakage

The native global candidate pool has a substantial shortcut: a model can often
identify the edited target from source audio alone without using the instruction.
A deterministic, non-neural 64-bin log-mel fingerprint baseline obtained the
following full-pool results:

| Diagnostic | Recall@1 | mAP@10 | nDCG@10 |
|---|---:|---:|---:|
| Random expectation | 0.0003 | 0.0009 | 0.0014 |
| Source-only spectral fingerprint | 0.5849 | 0.6094 | 0.6216 |
| Source duration + spectral fingerprint | 0.8113 | 0.8376 | 0.8495 |

Consequently, scores on this benchmark must not be interpreted as measuring
instruction use alone. Results should be accompanied by an audio-only ablation
where possible. The native 3,317-pair design is intentionally preserved rather
than introducing synthetic edits or reviewer-unapproved hard negatives.

## Source reuse and duplicate handling

The audit found five byte-identical source groups, four of which have multiple
distinct edited targets. It also found ten byte-identical target groups. The
corpus retains every native item, while qrels mark every byte-identical copy of
a query's target as relevant.

## License and citation

The source dataset is published under Apache-2.0. Please also cite the original
MMEdit work:

```bibtex
@article{tao2025mmedit,
  author = {Tao, Ye and Xu, Xuenan and Wu, Wen and Wang, Shuai and Wu, Mengyue and Zhang, Chao},
  journal = {arXiv preprint arXiv:2512.20339},
  title = {MMEDIT: A Unified Framework for Multi-Type Audio Editing via Audio Language Model},
  url = {https://arxiv.org/abs/2512.20339},
  year = {2025},
}
```
"""


@dataclass(frozen=True)
class AudioInfo:
    """Header and content-hash information for one WAV file."""

    audio_id: str
    path: Path
    size_bytes: int
    sha256: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frames: int
    duration_seconds: float


@dataclass
class AuditResult:
    """Machine-readable output from a complete source audit."""

    source_repo_id: str
    source_revision: str
    source_dir: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]
    edit_families: dict[str, int]
    audio_summary: dict[str, Any]
    duplicate_summary: dict[str, Any]
    source_reuse_summary: dict[str, Any]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_external_directory(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_relative_to(resolved, _REPO_ROOT):
        raise SystemExit(
            f"{description} must be outside the Git repository ({_REPO_ROOT}): "
            f"{resolved}"
        )
    return resolved


def _download_source(cache_dir: Path | None, *, offline: bool) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": SOURCE_REPO_ID,
        "repo_type": "dataset",
        "revision": SOURCE_REVISION,
        "allow_patterns": ["README.md", "content.jsonl", "raw/*.wav", "target/*.wav"],
        "local_files_only": offline,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    print(f"Source: {SOURCE_REPO_ID}@{SOURCE_REVISION}")
    print("Resolving pinned source snapshot...")
    return Path(snapshot_download(**kwargs)).resolve()


def _load_instructions(
    metadata_path: Path,
) -> tuple[dict[str, str], list[str], list[str]]:
    instructions: dict[str, str] = {}
    duplicate_ids: list[str] = []
    malformed_rows: list[str] = []

    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_rows.append(f"line {line_number}: invalid JSON ({error})")
                continue

            audio_id = row.get("audio_id")
            caption = row.get("caption")
            if not isinstance(audio_id, str) or not audio_id.strip():
                malformed_rows.append(f"line {line_number}: missing string audio_id")
                continue
            if not isinstance(caption, str) or not caption.strip():
                malformed_rows.append(
                    f"line {line_number}: {audio_id!r} has an empty caption"
                )
                continue
            if audio_id in instructions:
                duplicate_ids.append(audio_id)
                continue
            instructions[audio_id] = caption.strip()

    return instructions, duplicate_ids, malformed_rows


def _index_wavs(directory: Path) -> tuple[dict[str, Path], list[str]]:
    wavs: dict[str, Path] = {}
    duplicate_ids: list[str] = []
    for path in sorted(directory.glob("*.wav")):
        if path.stem in wavs:
            duplicate_ids.append(path.stem)
        else:
            wavs[path.stem] = path
    return wavs, duplicate_ids


def _inspect_wav(item: tuple[str, Path]) -> AudioInfo:
    audio_id, path = item
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    try:
        with wave_open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (EOFError, WaveError) as error:
        raise ValueError(f"{path}: cannot decode WAV header ({error})") from error

    if sample_rate <= 0:
        raise ValueError(f"{path}: invalid sample rate {sample_rate}")
    return AudioInfo(
        audio_id=audio_id,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frames=frames,
        duration_seconds=frames / sample_rate,
    )


def _inspect_all_wavs(
    wavs: dict[str, Path], *, workers: int, description: str
) -> tuple[dict[str, AudioInfo], list[str]]:
    infos: dict[str, AudioInfo] = {}
    errors: list[str] = []

    def inspect(item: tuple[str, Path]) -> tuple[str, AudioInfo | None, str | None]:
        audio_id, _ = item
        try:
            return audio_id, _inspect_wav(item), None
        except (OSError, ValueError) as error:
            return audio_id, None, str(error)

    items = sorted(wavs.items())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(inspect, items)
        for audio_id, info, error in tqdm(
            results, total=len(items), desc=description, unit="file"
        ):
            if error is not None:
                errors.append(error)
            elif info is not None:
                infos[audio_id] = info
    return infos, errors


def _group_by_hash(infos: dict[str, AudioInfo]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for audio_id, info in infos.items():
        groups[info.sha256].append(audio_id)
    return {digest: sorted(ids) for digest, ids in groups.items()}


def _duplicate_groups(groups: dict[str, list[str]]) -> list[list[str]]:
    return sorted(
        (ids for ids in groups.values() if len(ids) > 1),
        key=lambda ids: (-len(ids), ids),
    )


def _preview(values: list[str], limit: int = 8) -> str:
    shown = ", ".join(values[:limit])
    remaining = len(values) - limit
    return f"{shown}{f', +{remaining} more' if remaining > 0 else ''}"


def _validate_audio_properties(infos: dict[str, AudioInfo], *, label: str) -> list[str]:
    errors: list[str] = []
    bad_sample_rates = sorted(
        audio_id
        for audio_id, info in infos.items()
        if info.sample_rate != EXPECTED_SAMPLE_RATE
    )
    bad_channels = sorted(
        audio_id
        for audio_id, info in infos.items()
        if info.channels != EXPECTED_CHANNELS
    )
    bad_durations = sorted(
        audio_id
        for audio_id, info in infos.items()
        if abs(info.duration_seconds - EXPECTED_DURATION_SECONDS)
        > _DURATION_TOLERANCE_SECONDS
    )
    if bad_sample_rates:
        errors.append(
            f"{label}: {len(bad_sample_rates)} files are not "
            f"{EXPECTED_SAMPLE_RATE} Hz ({_preview(bad_sample_rates)})"
        )
    if bad_channels:
        errors.append(
            f"{label}: {len(bad_channels)} files are not mono "
            f"({_preview(bad_channels)})"
        )
    if bad_durations:
        errors.append(
            f"{label}: {len(bad_durations)} files differ from "
            f"{EXPECTED_DURATION_SECONDS:.1f}s by more than "
            f"{_DURATION_TOLERANCE_SECONDS:.2f}s ({_preview(bad_durations)})"
        )
    return errors


def audit_source(source_dir: Path, *, workers: int) -> AuditResult:
    """Run all structural, audio-format, hash, and reuse checks."""
    errors: list[str] = []
    warnings: list[str] = []
    metadata_path = source_dir / "content.jsonl"
    raw_dir = source_dir / "raw"
    target_dir = source_dir / "target"

    for required in (metadata_path, raw_dir, target_dir):
        if not required.exists():
            errors.append(f"Missing required source path: {required}")
    if errors:
        return AuditResult(
            source_repo_id=SOURCE_REPO_ID,
            source_revision=SOURCE_REVISION,
            source_dir=str(source_dir),
            valid=False,
            errors=errors,
            warnings=warnings,
            counts={},
            edit_families={},
            audio_summary={},
            duplicate_summary={},
            source_reuse_summary={},
        )

    instructions, duplicate_instruction_ids, malformed_rows = _load_instructions(
        metadata_path
    )
    raw_wavs, duplicate_raw_ids = _index_wavs(raw_dir)
    target_wavs, duplicate_target_ids = _index_wavs(target_dir)

    if malformed_rows:
        errors.extend(malformed_rows)
    if duplicate_instruction_ids:
        errors.append(
            f"Duplicate instruction IDs: {_preview(sorted(duplicate_instruction_ids))}"
        )
    if duplicate_raw_ids:
        errors.append(f"Duplicate raw IDs: {_preview(sorted(duplicate_raw_ids))}")
    if duplicate_target_ids:
        errors.append(f"Duplicate target IDs: {_preview(sorted(duplicate_target_ids))}")

    instruction_ids = set(instructions)
    raw_ids = set(raw_wavs)
    target_ids = set(target_wavs)
    all_ids = instruction_ids | raw_ids | target_ids
    complete_ids = instruction_ids & raw_ids & target_ids
    missing_sources = sorted((instruction_ids | target_ids) - raw_ids)
    missing_targets = sorted((instruction_ids | raw_ids) - target_ids)
    missing_instructions = sorted((raw_ids | target_ids) - instruction_ids)

    for label, values in (
        ("Missing sources", missing_sources),
        ("Missing targets", missing_targets),
        ("Missing instructions", missing_instructions),
    ):
        if values:
            errors.append(f"{label}: {_preview(values)}")

    expected_counts = {
        "instructions": len(instructions),
        "raw_wavs": len(raw_wavs),
        "target_wavs": len(target_wavs),
        "complete_triplets": len(complete_ids),
    }
    for label, count in expected_counts.items():
        if count != EXPECTED_TRIPLETS:
            errors.append(f"Expected {EXPECTED_TRIPLETS} {label}, found {count}")

    family_counts: Counter[str] = Counter()
    invalid_family_ids: list[str] = []
    for audio_id in sorted(all_ids):
        match = _AUDIO_ID_RE.fullmatch(audio_id)
        if match is None:
            invalid_family_ids.append(audio_id)
        else:
            family_counts[match.group("family")] += 1
    if invalid_family_ids:
        errors.append(
            "IDs without a '<family>_<number>' structure: "
            f"{_preview(invalid_family_ids)}"
        )

    raw_infos, raw_decode_errors = _inspect_all_wavs(
        raw_wavs, workers=workers, description="Audit raw audio"
    )
    target_infos, target_decode_errors = _inspect_all_wavs(
        target_wavs, workers=workers, description="Audit target audio"
    )
    errors.extend(raw_decode_errors)
    errors.extend(target_decode_errors)
    # The source card's format claims do not match every file, but the audio is
    # decodable and usable. Preserve these discrepancies as prominent warnings;
    # structural failures and undecodable files remain blocking errors.
    warnings.extend(_validate_audio_properties(raw_infos, label="raw"))
    warnings.extend(_validate_audio_properties(target_infos, label="target"))

    raw_hash_groups = _group_by_hash(raw_infos)
    target_hash_groups = _group_by_hash(target_infos)
    raw_duplicate_groups = _duplicate_groups(raw_hash_groups)
    target_duplicate_groups = _duplicate_groups(target_hash_groups)
    shared_hashes = set(raw_hash_groups) & set(target_hash_groups)
    identical_pairs = sorted(
        audio_id
        for audio_id in complete_ids
        if audio_id in raw_infos
        and audio_id in target_infos
        and raw_infos[audio_id].sha256 == target_infos[audio_id].sha256
    )
    if identical_pairs:
        errors.append(
            f"{len(identical_pairs)} triplets have byte-identical source and target "
            f"audio ({_preview(identical_pairs)})"
        )
    if raw_duplicate_groups:
        warnings.append(
            f"Raw audio has {len(raw_duplicate_groups)} byte-identical duplicate "
            "groups; inspect source reuse before defining qrels"
        )
    if target_duplicate_groups:
        warnings.append(
            f"Target audio has {len(target_duplicate_groups)} byte-identical duplicate "
            "groups; relevant duplicates may need additional qrels"
        )

    reused_sources: list[dict[str, Any]] = []
    for digest, ids in raw_hash_groups.items():
        if len(ids) < 2:
            continue
        captions = {
            instructions[audio_id] for audio_id in ids if audio_id in instructions
        }
        target_hashes = {
            target_infos[audio_id].sha256
            for audio_id in ids
            if audio_id in target_infos
        }
        reused_sources.append(
            {
                "sha256": digest,
                "audio_ids": ids,
                "instruction_count": len(captions),
                "target_count": len(target_hashes),
            }
        )
    reused_sources.sort(
        key=lambda group: (-len(group["audio_ids"]), group["audio_ids"])
    )

    durations = [
        info.duration_seconds for info in [*raw_infos.values(), *target_infos.values()]
    ]
    sampling_rates = Counter(
        info.sample_rate for info in [*raw_infos.values(), *target_infos.values()]
    )
    channels = Counter(
        info.channels for info in [*raw_infos.values(), *target_infos.values()]
    )
    sample_widths = Counter(
        info.sample_width_bytes
        for info in [*raw_infos.values(), *target_infos.values()]
    )
    total_size_bytes = sum(
        info.size_bytes for info in [*raw_infos.values(), *target_infos.values()]
    )

    duplicate_summary = {
        "raw_unique_hashes": len(raw_hash_groups),
        "target_unique_hashes": len(target_hash_groups),
        "raw_duplicate_groups": len(raw_duplicate_groups),
        "raw_duplicate_files": sum(len(ids) for ids in raw_duplicate_groups),
        "target_duplicate_groups": len(target_duplicate_groups),
        "target_duplicate_files": sum(len(ids) for ids in target_duplicate_groups),
        "source_target_shared_hashes": len(shared_hashes),
        "identical_source_target_pairs": len(identical_pairs),
        "raw_duplicate_examples": raw_duplicate_groups[:10],
        "target_duplicate_examples": target_duplicate_groups[:10],
    }
    source_reuse_summary = {
        "reused_source_groups": len(reused_sources),
        "groups_with_multiple_instructions": sum(
            group["instruction_count"] > 1 for group in reused_sources
        ),
        "groups_with_multiple_targets": sum(
            group["target_count"] > 1 for group in reused_sources
        ),
        "examples": reused_sources[:10],
    }
    audio_summary = {
        "decoded_files": len(raw_infos) + len(target_infos),
        "total_size_bytes": total_size_bytes,
        "duration_seconds": {
            "minimum": min(durations) if durations else None,
            "average": sum(durations) / len(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "sampling_rates": dict(sorted(sampling_rates.items())),
        "channels": dict(sorted(channels.items())),
        "sample_width_bytes": dict(sorted(sample_widths.items())),
    }
    counts = {
        **expected_counts,
        "all_ids": len(all_ids),
        "edit_families": len(family_counts),
    }

    return AuditResult(
        source_repo_id=SOURCE_REPO_ID,
        source_revision=SOURCE_REVISION,
        source_dir=str(source_dir),
        valid=not errors,
        errors=errors,
        warnings=warnings,
        counts=counts,
        edit_families=dict(sorted(family_counts.items())),
        audio_summary=audio_summary,
        duplicate_summary=duplicate_summary,
        source_reuse_summary=source_reuse_summary,
    )


def _build_retrieval_data(
    source_dir: Path, *, workers: int
) -> tuple[dict[str, DatasetDict], dict[str, Any]]:
    """Construct deterministic MTEB queries, corpus, and duplicate-aware qrels."""
    instructions, duplicate_ids, malformed_rows = _load_instructions(
        source_dir / "content.jsonl"
    )
    raw_wavs, duplicate_raw_ids = _index_wavs(source_dir / "raw")
    target_wavs, duplicate_target_ids = _index_wavs(source_dir / "target")
    if duplicate_ids or malformed_rows or duplicate_raw_ids or duplicate_target_ids:
        raise ValueError("Source metadata changed after audit; refusing to build")

    complete_ids = sorted(set(instructions) & set(raw_wavs) & set(target_wavs))
    if len(complete_ids) != EXPECTED_TRIPLETS:
        raise ValueError(
            f"Expected {EXPECTED_TRIPLETS:,} complete triplets, "
            f"found {len(complete_ids):,}"
        )

    target_infos, decode_errors = _inspect_all_wavs(
        target_wavs,
        workers=workers,
        description="Hash target audio for qrels",
    )
    if decode_errors or len(target_infos) != EXPECTED_TRIPLETS:
        raise ValueError(
            "Target audio changed after audit; refusing to build: "
            f"{_preview(decode_errors)}"
        )
    target_hash_groups = _group_by_hash(target_infos)

    query_rows = {
        "id": [f"q-{audio_id}" for audio_id in complete_ids],
        "audio": [str(raw_wavs[audio_id]) for audio_id in complete_ids],
        "text": [instructions[audio_id] for audio_id in complete_ids],
    }
    corpus_rows = {
        "id": [f"t-{audio_id}" for audio_id in complete_ids],
        "audio": [str(target_wavs[audio_id]) for audio_id in complete_ids],
    }
    qrel_rows: dict[str, list[Any]] = {
        "query-id": [],
        "corpus-id": [],
        "score": [],
    }
    for audio_id in complete_ids:
        relevant_ids = target_hash_groups[target_infos[audio_id].sha256]
        for relevant_id in relevant_ids:
            qrel_rows["query-id"].append(f"q-{audio_id}")
            qrel_rows["corpus-id"].append(f"t-{relevant_id}")
            qrel_rows["score"].append(1)

    query_ids = set(query_rows["id"])
    corpus_ids = set(corpus_rows["id"])
    qrel_query_ids = set(qrel_rows["query-id"])
    qrel_corpus_ids = set(qrel_rows["corpus-id"])
    if len(query_ids) != len(query_rows["id"]):
        raise ValueError("Constructed duplicate query IDs")
    if len(corpus_ids) != len(corpus_rows["id"]):
        raise ValueError("Constructed duplicate corpus IDs")
    if qrel_query_ids != query_ids:
        raise ValueError("Not every query has at least one positive qrel")
    if not qrel_corpus_ids <= corpus_ids:
        raise ValueError("A qrel references a missing corpus ID")
    if len(qrel_rows["query-id"]) != EXPECTED_QRELS:
        raise ValueError(
            f"Expected {EXPECTED_QRELS:,} duplicate-aware qrels, "
            f"constructed {len(qrel_rows['query-id']):,}"
        )

    # Cast after constructing the Arrow table. ``datasets>=5`` otherwise tries
    # to re-encode every WAV through the optional torchcodec dependency even
    # though these rows already point to valid source files.
    queries = Dataset.from_dict(query_rows).cast_column("audio", Audio())
    corpus = Dataset.from_dict(corpus_rows).cast_column("audio", Audio())
    qrels = Dataset.from_dict(
        qrel_rows,
        features=Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("int32"),
            }
        ),
    )
    datasets = {
        "queries": DatasetDict({"test": queries}),
        "corpus": DatasetDict({"test": corpus}),
        "qrels": DatasetDict({"test": qrels}),
    }
    manifest = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "counts": {
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": len(qrels),
        },
        "id_format": {
            "query": "q-{audio_id}",
            "corpus": "t-{audio_id}",
        },
        "duplicate_qrels_policy": (
            "Every corpus item with a byte-identical target WAV is relevant."
        ),
        "target_duplicate_groups": _duplicate_groups(target_hash_groups),
        "audio_transformation": "none",
        "source_only_diagnostic": {
            "method": "64-bin log-mel distribution fingerprint",
            "spectral_fingerprint": {
                "recall_at_1": 0.5848658426288815,
                "map_at_10": 0.6094354001272904,
                "ndcg_at_10": 0.6215856571177805,
            },
            "duration_then_fingerprint": {
                "recall_at_1": 0.8112752487187217,
                "map_at_10": 0.8376390743213172,
                "ndcg_at_10": 0.8495169772023567,
            },
            "random_expected": {
                "recall_at_1": 0.0003032950089747874,
                "map_at_10": 0.0008830211369784203,
                "ndcg_at_10": 0.0013716487209221259,
            },
        },
    }
    return datasets, manifest


def _write_local_dataset(
    datasets: dict[str, DatasetDict], manifest: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [
        name
        for name in (*datasets, "README.md", "manifest.json")
        if (output_dir / name).exists()
    ]
    if occupied:
        raise ValueError(
            f"Output directory is not empty ({_preview(occupied)}); "
            "choose a new --output-dir"
        )
    for config_name, dataset in datasets.items():
        dataset.save_to_disk(output_dir / config_name)
    (output_dir / "README.md").write_text(_DATASET_CARD, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote local MTEB dataset: {output_dir}")


def _push_dataset(datasets: dict[str, DatasetDict], *, repo_id: str, token: str) -> str:
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    for config_name in ("queries", "corpus", "qrels"):
        print(f"Pushing {config_name} configuration to {repo_id}...")
        datasets[config_name].push_to_hub(
            repo_id,
            config_name=config_name,
            token=token,
            max_shard_size="500MB",
        )
    HfApi(token=token).upload_file(
        path_or_fileobj=_DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document MMEdit-AT2A construction and limitations",
    )
    revision = dataset_info(repo_id, token=token).sha
    print(f"Pushed {repo_id}@{revision}")
    return revision


def _print_summary(result: AuditResult) -> None:
    print("\nMMEdit source audit")
    print(f"  Revision:          {result.source_revision}")
    print(f"  Snapshot:          {result.source_dir}")
    if result.counts:
        print(f"  Complete triplets: {result.counts['complete_triplets']:,}")
        print(f"  Edit families:     {result.counts['edit_families']}")
    if result.audio_summary:
        duration = result.audio_summary["duration_seconds"]
        total_gib = result.audio_summary["total_size_bytes"] / 1024**3
        print(f"  Decoded WAVs:      {result.audio_summary['decoded_files']:,}")
        print(f"  Audio size:        {total_gib:.2f} GiB")
        if duration["minimum"] is not None and duration["maximum"] is not None:
            print(
                "  Duration range:    "
                f"{duration['minimum']:.3f}s–{duration['maximum']:.3f}s"
            )
    if result.duplicate_summary:
        duplicates = result.duplicate_summary
        print(
            "  Duplicate groups:  "
            f"raw={duplicates['raw_duplicate_groups']}, "
            f"target={duplicates['target_duplicate_groups']}"
        )
        print(
            "  Shared hashes:     "
            f"{duplicates['source_target_shared_hashes']} source↔target"
        )
        print(f"  Identical pairs:   {duplicates['identical_source_target_pairs']}")
    if result.source_reuse_summary:
        reuse = result.source_reuse_summary
        print(
            "  Reused sources:    "
            f"{reuse['reused_source_groups']} groups "
            f"({reuse['groups_with_multiple_targets']} with multiple targets)"
        )
    if result.edit_families:
        families = ", ".join(
            f"{family}={count}" for family, count in result.edit_families.items()
        )
        print(f"  Family counts:     {families}")
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"  Result:            {'PASS' if result.valid else 'FAIL'}")


def _json_ready(result: AuditResult) -> dict[str, Any]:
    data = asdict(result)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Audit an existing snapshot instead of downloading it.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory; must be outside this Git repository.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached/local files only; never access the network.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, (os.cpu_count() or 1) * 2)),
        help="Threads used for WAV inspection and hashing.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optionally write the complete audit report as JSON.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Construct the deterministic MTEB queries/corpus/qrels after auditing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write a local dataset export here; must be outside this repository.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_OUTPUT_REPO_ID,
        help=f"Hugging Face output repository (default: {DEFAULT_OUTPUT_REPO_ID}).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload all three configurations. Requires --build and HF_TOKEN.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.source_dir is not None and args.cache_dir is not None:
        parser.error("--source-dir and --cache-dir cannot be used together")
    if args.output_dir is not None and not args.build:
        parser.error("--output-dir requires --build")
    if args.push and not args.build:
        parser.error("--push requires --build")

    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = _require_external_directory(
            args.cache_dir, description="Cache directory"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

    if args.source_dir is not None:
        source_dir = _require_external_directory(
            args.source_dir, description="Source directory"
        )
    else:
        source_dir = _download_source(cache_dir, offline=args.offline)

    result = audit_source(source_dir, workers=args.workers)
    _print_summary(result)

    if args.json_report is not None:
        report_path = args.json_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {report_path}")

    if not result.valid:
        raise SystemExit(1)
    if not args.build:
        return

    try:
        datasets, manifest = _build_retrieval_data(source_dir, workers=args.workers)
        if args.output_dir is not None:
            output_dir = _require_external_directory(
                args.output_dir, description="Output directory"
            )
            _write_local_dataset(datasets, manifest, output_dir)
        if args.push:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("Set HF_TOKEN in the environment before using --push")
            revision = _push_dataset(datasets, repo_id=args.repo_id, token=token)
            if args.output_dir is not None:
                (output_dir / "hub_revision.txt").write_text(
                    revision + "\n", encoding="utf-8"
                )
        elif args.output_dir is None:
            print(
                "Build validated in memory; use --output-dir to save or --push to upload."
            )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
