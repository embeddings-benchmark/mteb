#!/usr/bin/env python3
"""Build the MTEB release of SpanishWikiClustering v2.

By default the script downloads the public, immutable source release
``ClementeH/SpanishWikiClustering-v2`` and reconstructs a Hub-ready MTEB
artifact from it.  Thus the ordinary transformation is reproducible by anyone
without access to local construction artifacts.

The optional ``--input`` mode starts one step earlier, from the audited
``final_pool_v1.jsonl``.  That source pool is produced from the pinned
``eswiki-20260801`` snapshot with the public construction pipeline at
https://github.com/Clemente-H/spanish-wiki-clustering-v2.  It is intentionally
not checked into that repository: it contains redistributed Wikipedia text and
the published Hub release is the distribution point for that material.

Both modes validate the frozen 1,200-item, six-way balanced evaluation split
and write:

  data/test.parquet                 ``sentences: string``, ``labels: int64``
  provenance/provenance.parquet     attribution and category evidence
  label_mapping.json
  transformation_manifest.json

Usage:
  # Rebuild from the public pinned source release (no remote write).
  uv run python scripts/data/spanish_wiki_clustering_v2/create_data.py \\
      --output-dir /tmp/spanish_wiki_clustering_v2

  # Or transform the earlier audited source pool.
  uv run python scripts/data/spanish_wiki_clustering_v2/create_data.py \\
      --input /path/to/final_pool_v1.jsonl \\
      --output-dir /tmp/spanish_wiki_clustering_v2

  # Upload only after inspecting the local output.
  export HF_TOKEN=...
  uv run python scripts/data/spanish_wiki_clustering_v2/create_data.py \\
      --output-dir /tmp/spanish_wiki_clustering_v2 \\
      --repo-id your-namespace/SpanishWikiClustering-v2 --push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download


LABELS = (
    "Artes y medios",
    "Deporte",
    "Historia y arqueología",
    "Matemáticas y computación",
    "Medicina clínica",
    "Política, derecho y sociedad",
)
EXPECTED_RECORDS = 1_200
EXPECTED_PER_LABEL = 200
SOURCE_REPO = "ClementeH/SpanishWikiClustering-v2"
SOURCE_REVISION = "f23479438e369c36dda6b54211824b87d7010213"
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "article_id",
        "title",
        "text",
        "label",
        "category_evidence",
        "source_revision",
    }
)


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pool(path: Path) -> list[dict[str, Any]]:
    """Read and validate the canonical, audited final-pool JSONL."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_SOURCE_COLUMNS.difference(record)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required columns: {sorted(missing)}"
                )
            records.append(record)

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} records, found {len(records)}")
    if len({record["article_id"] for record in records}) != len(records):
        raise ValueError("The source pool contains duplicate article IDs")
    if any(
        not isinstance(record["text"], str) or not record["text"].strip()
        for record in records
    ):
        raise ValueError("The source pool contains an empty text")
    if len({record["text"] for record in records}) != len(records):
        raise ValueError("The source pool contains duplicate texts")

    label_counts = Counter(record["label"] for record in records)
    expected_counts = Counter({label: EXPECTED_PER_LABEL for label in LABELS})
    if label_counts != expected_counts:
        raise ValueError(
            f"Unexpected label counts: {dict(sorted(label_counts.items()))}"
        )
    return sorted(records, key=lambda record: str(record["article_id"]))


def permalink(title: str, source_revision: str) -> str:
    """Construct a Spanish Wikipedia permalink for the recorded revision."""
    revision = source_revision.rsplit("rev=", maxsplit=1)[-1]
    if not revision.isdigit():
        raise ValueError(f"Could not extract an oldid from {source_revision!r}")
    return "https://es.wikipedia.org/w/index.php?title={}&oldid={}".format(
        quote(title.replace(" ", "_")), revision
    )


def build_dataset(records: list[dict[str, Any]]) -> tuple[DatasetDict, Dataset]:
    """Convert canonical records to the evaluation and provenance tables."""
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    evaluation = Dataset.from_list(
        [
            {"sentences": record["text"], "labels": label_to_id[record["label"]]}
            for record in records
        ]
    )
    provenance = Dataset.from_list(
        [
            {
                "article_id": str(record["article_id"]),
                "title": record["title"],
                "source_revision": record["source_revision"],
                "permalink": permalink(record["title"], record["source_revision"]),
                "category_evidence": json.dumps(
                    record["category_evidence"], ensure_ascii=False, sort_keys=True
                ),
                "extraction_rule": record.get(
                    "extraction_rule",
                    "final-pool-v1/depth=1/exclusive/evidence-preserved",
                ),
            }
            for record in records
        ]
    )
    return DatasetDict({"test": evaluation}), provenance


def validate_evaluation(dataset: Dataset) -> None:
    """Validate the public MTEB evaluation-table contract."""
    if dataset.column_names != ["sentences", "labels"]:
        raise ValueError(f"Unexpected evaluation columns: {dataset.column_names}")
    if len(dataset) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} records, found {len(dataset)}")
    if any(not text.strip() for text in dataset["sentences"]):
        raise ValueError("The evaluation split contains an empty text")
    if len(set(dataset["sentences"])) != EXPECTED_RECORDS:
        raise ValueError("The evaluation split contains duplicate texts")
    expected_counts = Counter(
        {index: EXPECTED_PER_LABEL for index in range(len(LABELS))}
    )
    if Counter(dataset["labels"]) != expected_counts:
        raise ValueError("The evaluation split is not balanced 6 × 200")


def build_from_published_source(
    repo_id: str, revision: str
) -> tuple[DatasetDict, Path, Path]:
    """Load the immutable public release and its provenance sidecars."""
    source = load_dataset(repo_id, revision=revision, split="test")
    validate_evaluation(source)
    provenance_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename="provenance/provenance.parquet",
        )
    )
    mapping_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename="label_mapping.json",
        )
    )
    expected_mapping = {str(index): label for index, label in enumerate(LABELS)}
    if json.loads(mapping_path.read_text(encoding="utf-8")) != expected_mapping:
        raise ValueError("The public label mapping does not match the frozen ontology")
    return DatasetDict({"test": source}), provenance_path, mapping_path


def write_output(
    dataset: DatasetDict, provenance: Dataset, input_path: Path, output_dir: Path
) -> None:
    """Write the Hub-ready release files without performing a remote write."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "provenance").mkdir()
    dataset["test"].to_parquet(output_dir / "data" / "test.parquet")
    provenance.to_parquet(output_dir / "provenance" / "provenance.parquet")
    mapping = {str(index): label for index, label in enumerate(LABELS)}
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_snapshot": "eswiki-20260801",
        "input": input_path.name,
        "input_sha256": sha256(input_path),
        "records": EXPECTED_RECORDS,
        "split": "test",
        "schema": {"sentences": "string", "labels": "int64"},
        "label_mapping": mapping,
        "release_files": {
            "data/test.parquet": sha256(output_dir / "data" / "test.parquet"),
            "provenance/provenance.parquet": sha256(
                output_dir / "provenance" / "provenance.parquet"
            ),
        },
    }
    (output_dir / "transformation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_published_output(
    dataset: DatasetDict,
    provenance_path: Path,
    mapping_path: Path,
    repo_id: str,
    revision: str,
    output_dir: Path,
) -> None:
    """Write a local MTEB artifact from a pinned public source release."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "provenance").mkdir()
    dataset["test"].to_parquet(output_dir / "data" / "test.parquet")
    shutil.copy2(provenance_path, output_dir / "provenance" / "provenance.parquet")
    shutil.copy2(mapping_path, output_dir / "label_mapping.json")
    manifest = {
        "source_repo": repo_id,
        "source_revision": revision,
        "records": EXPECTED_RECORDS,
        "split": "test",
        "schema": {"sentences": "string", "labels": "int64"},
        "release_files": {
            "data/test.parquet": sha256(output_dir / "data" / "test.parquet"),
            "provenance/provenance.parquet": sha256(
                output_dir / "provenance" / "provenance.parquet"
            ),
        },
    }
    (output_dir / "transformation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional audited final-pool JSONL; omit to use the public source release",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New release directory"
    )
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--repo-id", help="Optional Hugging Face dataset repository")
    parser.add_argument(
        "--push", action="store_true", help="Upload the inspected output"
    )
    args = parser.parse_args()

    if args.push and not args.repo_id:
        parser.error("--push requires --repo-id")
    if args.input is not None and not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")

    if args.input is None:
        dataset, provenance_path, mapping_path = build_from_published_source(
            args.source_repo, args.source_revision
        )
        write_published_output(
            dataset,
            provenance_path,
            mapping_path,
            args.source_repo,
            args.source_revision,
            args.output_dir,
        )
    else:
        records = read_pool(args.input)
        dataset, provenance = build_dataset(records)
        write_output(dataset, provenance, args.input, args.output_dir)
    validate_evaluation(dataset["test"])
    print(
        json.dumps(
            {
                "records": len(dataset["test"]),
                "label_counts": Counter(dataset["test"]["labels"]),
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            default=dict,
        )
    )

    if args.push:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=args.output_dir,
            commit_message="Build SpanishWikiClustering v2 from audited source pool",
        )
        print(f"Uploaded release to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
