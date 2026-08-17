"""Validate the pinned MixBench payloads and original relevance judgments."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

MIXBENCH_REPO = "mixed-modality-search/MixBench2026"
MIXBENCH_REVISION = "17a9e705b2346b118a63f163f10e47325f9e9ecc"
QRELS_REPO = "mixed-modality-search/MixBench25"
QRELS_REVISION = "88e3916036ea0bdb205f4da885d6e947a565c1a0"
CONFIGS = ("MSCOCO", "Google_WIT", "VisualNews", "OVEN")
EXPECTED = {
    "MSCOCO": {
        "queries": 984,
        "original_corpus": 984,
        "mixed_corpus": 984,
        "qrels": 984,
        "query_modalities": {"text": 984},
        "corpus_modalities": {"image+text": 328, "image": 328, "text": 328},
    },
    "Google_WIT": {
        "queries": 1000,
        "original_corpus": 4421,
        "mixed_corpus": 4421,
        "qrels": 1000,
        "query_modalities": {"text": 1000},
        "corpus_modalities": {"image+text": 1475, "image": 1473, "text": 1473},
    },
    "VisualNews": {
        "queries": 981,
        "original_corpus": 981,
        "mixed_corpus": 981,
        "qrels": 981,
        "query_modalities": {"text": 981},
        "corpus_modalities": {"image+text": 327, "image": 327, "text": 327},
    },
    "OVEN": {
        "queries": 1000,
        "original_corpus": 1000,
        "mixed_corpus": 1000,
        "qrels": 1000,
        "query_modalities": {"image+text": 1000},
        "corpus_modalities": {"image+text": 334, "image": 333, "text": 333},
    },
}


def _modality(text: str | None, image: dict[str, Any] | None) -> str:
    parts = []
    if image is not None:
        parts.append("image")
    if text:
        parts.append("text")
    if not parts:
        raise AssertionError("Found a row without text or image")
    return "+".join(parts)


def _download_sources() -> tuple[Path, Path]:
    current = Path(
        snapshot_download(
            repo_id=MIXBENCH_REPO,
            repo_type="dataset",
            revision=MIXBENCH_REVISION,
            allow_patterns=[
                "README.md",
                ".dataset_config.json",
                "*/queries.parquet",
                "*/mixed_corpus.parquet",
            ],
        )
    )
    original = Path(
        snapshot_download(
            repo_id=QRELS_REPO,
            repo_type="dataset",
            revision=QRELS_REVISION,
            allow_patterns=[
                "README.md",
                "MixBench25.py",
                "*/queries.jsonl",
                "*/corpus.jsonl",
                "*/mixed_corpus.jsonl",
                "*/qrels/qrels.tsv",
            ],
        )
    )
    return current, original


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _audit_parquet(
    path: Path,
    original_rows: list[dict[str, Any]],
    id_key: str,
) -> dict[str, Any]:
    original = {str(row[id_key]): row for row in original_rows}
    assert len(original) == len(original_rows), f"Duplicate IDs in {path} source JSONL"

    ids: list[str] = []
    modalities: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    mismatched_text = 0
    mismatched_image_presence = 0

    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            row_id = str(row["id"])
            ids.append(row_id)
            image = row.get("image")
            modality = _modality(row.get("text"), image)
            modalities[modality] += 1
            source_row = original.get(row_id)
            assert source_row is not None, f"{path}: ID {row_id} missing in MixBench25"
            mismatched_text += row.get("text") != source_row.get("text")
            mismatched_image_presence += (image is not None) != bool(
                source_row.get("image")
            )

            if image is not None:
                assert image.get("bytes") is not None, (
                    f"{path}: image bytes missing for {row_id}"
                )
                with Image.open(io.BytesIO(image["bytes"])) as decoded:
                    formats[str(decoded.format)] += 1
                    decoded.verify()

            if len(examples[modality]) < 2:
                examples[modality].append(
                    {"id": row_id, "text": (row.get("text") or "")[:120]}
                )

    assert len(ids) == len(set(ids)), f"Duplicate IDs in {path}"
    assert set(ids) == set(original), f"ID set differs between releases for {path}"
    assert mismatched_text == 0, f"Text differs between releases for {path}"
    assert mismatched_image_presence == 0, (
        f"Modality presence differs between releases for {path}"
    )
    return {
        "count": len(ids),
        "ids": set(ids),
        "modalities": dict(modalities),
        "image_formats": dict(formats),
        "examples": dict(examples),
    }


def _audit_qrels(
    path: Path, query_ids: set[str], corpus_ids: set[str]
) -> dict[str, Any]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    triples = [
        (str(row["query_id"]), str(row["corpus_id"]), int(row["score"])) for row in rows
    ]
    pairs = [(query_id, corpus_id) for query_id, corpus_id, _ in triples]
    per_query = Counter(query_id for query_id, _, _ in triples)

    assert len(pairs) == len(set(pairs)), f"Duplicate qrel pairs in {path}"
    assert set(per_query) == query_ids, f"Qrel query IDs do not resolve in {path}"
    assert {corpus_id for _, corpus_id, _ in triples} <= corpus_ids, (
        f"Qrel corpus IDs do not resolve in {path}"
    )
    assert all(score == 1 for _, _, score in triples), (
        f"Unexpected qrel score in {path}"
    )
    assert all(count >= 1 for count in per_query.values()), (
        f"A query has no positive in {path}"
    )
    return {
        "count": len(triples),
        "positives_per_query": dict(Counter(per_query.values())),
        "scores": dict(Counter(score for _, _, score in triples)),
        "identical_id_pairs": sum(
            query_id == corpus_id for query_id, corpus_id in pairs
        ),
    }


def validate(current: Path, original: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source": {"repo": MIXBENCH_REPO, "revision": MIXBENCH_REVISION},
        "qrels_source": {"repo": QRELS_REPO, "revision": QRELS_REVISION},
        "configs": {},
    }
    for config in CONFIGS:
        queries_source = _read_jsonl(original / config / "queries.jsonl")
        corpus_source = _read_jsonl(original / config / "mixed_corpus.jsonl")
        original_corpus = _read_jsonl(original / config / "corpus.jsonl")
        original_corpus_ids = [str(row["corpus_id"]) for row in original_corpus]
        assert len(original_corpus_ids) == len(set(original_corpus_ids)), (
            f"Duplicate original corpus IDs in {config}"
        )

        queries = _audit_parquet(
            current / config / "queries.parquet", queries_source, "query_id"
        )
        corpus = _audit_parquet(
            current / config / "mixed_corpus.parquet", corpus_source, "corpus_id"
        )
        qrels = _audit_qrels(
            original / config / "qrels" / "qrels.tsv",
            queries["ids"],
            corpus["ids"],
        )

        actual = {
            "queries": queries["count"],
            "original_corpus": len(original_corpus_ids),
            "mixed_corpus": corpus["count"],
            "qrels": qrels["count"],
            "query_modalities": queries["modalities"],
            "corpus_modalities": corpus["modalities"],
        }
        assert actual == EXPECTED[config], f"Unexpected counts for {config}: {actual}"
        report["configs"][config] = {
            **actual,
            "positives_per_query": qrels["positives_per_query"],
            "qrel_scores": qrels["scores"],
            "identical_id_qrel_pairs": qrels["identical_id_pairs"],
            "query_image_formats": queries["image_formats"],
            "corpus_image_formats": corpus["image_formats"],
            "representative_examples": {
                "queries": queries["examples"],
                "corpus": corpus["examples"],
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixbench-2026-dir", type=Path)
    parser.add_argument("--mixbench-25-dir", type=Path)
    args = parser.parse_args()
    if (args.mixbench_2026_dir is None) != (args.mixbench_25_dir is None):
        parser.error("Pass both local source directories or neither")
    current, original = (
        (args.mixbench_2026_dir, args.mixbench_25_dir)
        if args.mixbench_2026_dir is not None
        else _download_sources()
    )
    print(json.dumps(validate(current, original), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
