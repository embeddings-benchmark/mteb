#!/usr/bin/env python3
"""
Data preparation script for the SpeechEditBench Acoustic Editing retrieval task.
Downloads the SpeechEditBench dataset, processes the acoustic_editing subset,
creates MTEB retrieval splits (queries, corpus, qrels), and uploads to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import create_repo, dataset_info, snapshot_download


def process_and_upload(
    repo_id: str,
    token: str | None,
    push: bool,
    output_dir: Path | None,
) -> None:
    # 1. Download SpeechEditBench snapshot (acoustic_editing folder)
    print("Downloading SpeechEditBench acoustic editing files from HF Hub...")
    snapshot_path_str = snapshot_download(
        repo_id="DiscreteSpeech/SpeechEditBench",
        repo_type="dataset",
        allow_patterns="data/acoustic_editing/**",
        token=token,
    )
    snapshot_path = Path(snapshot_path_str)
    print(f"Snapshot downloaded to {snapshot_path}")

    # 2. Read samples.jsonl
    samples_file = snapshot_path / "data" / "acoustic_editing" / "samples.jsonl"
    if not samples_file.exists():
        raise FileNotFoundError(f"Could not find samples.jsonl at {samples_file}")

    print("Loading samples.jsonl...")
    samples = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from SpeechEditBench acoustic_editing.")

    # 3. Construct retrieval rows
    query_rows: dict[str, list[Any]] = {
        "id": [],
        "audio": [],
        "text": [],
    }
    corpus_rows: dict[str, list[Any]] = {
        "id": [],
        "audio": [],
    }
    qrel_rows: dict[str, list[Any]] = {
        "query-id": [],
        "corpus-id": [],
        "score": [],
    }

    for s in samples:
        sample_id = s["sample_id"]
        instruction = s["instruction"]
        source_rel_path = s["audio_path"]
        target_rel_path = s["anchor"]["target_reference_path"]

        source_abs_path = snapshot_path / source_rel_path
        target_abs_path = snapshot_path / target_rel_path

        if not source_abs_path.exists():
            print(f"Warning: Source audio file not found: {source_abs_path}")
            continue
        if not target_abs_path.exists():
            print(f"Warning: Target audio file not found: {target_abs_path}")
            continue

        query_id = f"q-{sample_id}"
        corpus_id = f"t-{sample_id}"

        query_rows["id"].append(query_id)
        query_rows["audio"].append(str(source_abs_path))
        query_rows["text"].append(instruction)

        corpus_rows["id"].append(corpus_id)
        corpus_rows["audio"].append(str(target_abs_path))

        qrel_rows["query-id"].append(query_id)
        qrel_rows["corpus-id"].append(corpus_id)
        qrel_rows["score"].append(1)

    print(f"Constructed {len(query_rows['id'])} queries and {len(corpus_rows['id'])} corpus items.")

    # 4. Create Hugging Face Datasets
    print("Casting columns to Arrow schemas and Audio features...")
    queries_ds = Dataset.from_dict(query_rows).cast_column("audio", Audio())
    corpus_ds = Dataset.from_dict(corpus_rows).cast_column("audio", Audio())
    qrels_ds = Dataset.from_dict(
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
        "queries": DatasetDict({"test": queries_ds}),
        "corpus": DatasetDict({"test": corpus_ds}),
        "qrels": DatasetDict({"test": qrels_ds}),
    }

    # 5. Output / Upload
    if push:
        if not token:
            raise ValueError("Hugging Face Hub token (token or HF_TOKEN env var) must be provided to push.")
        print(f"Creating repository {repo_id} (or checking existence)...")
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

        for config_name, dataset_dict in datasets.items():
            print(f"Pushing {config_name} configuration to {repo_id}...")
            dataset_dict.push_to_hub(
                repo_id,
                config_name=config_name,
                token=token,
                max_shard_size="500MB",
            )

        readme_content = f"""---
license: cc-by-4.0
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

# SpeechEdit Acoustic Retrieval Dataset

This dataset is an MTEB-formatted Any-to-Any (AT2A) composed audio retrieval adaptation of the `acoustic_editing` subset of [`DiscreteSpeech/SpeechEditBench`](https://huggingface.co/datasets/DiscreteSpeech/SpeechEditBench).

Each query combines an original/source speech recording with a natural-language editing instruction, and the corpus contains the corresponding edited target speech recordings.

## Schema

- **queries**: `id` (string), `audio` (source audio), and `text` (edit instruction)
- **corpus**: `id` (string) and `audio` (edited target)
- **qrels**: `query-id` (string), `corpus-id` (string), and binary `score` (int32)
"""
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add MTEB SpeechEdit Acoustic metadata and README",
        )

        sha = dataset_info(repo_id, token=token).sha
        print(f"\nSuccessfully pushed dataset to Hub!\nRepo ID: {repo_id}\nRevision (SHA): {sha}\n")
    else:
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for config_name, dataset_dict in datasets.items():
                dataset_dict.save_to_disk(output_dir / config_name)
            print(f"Saved dataset locally to {output_dir}")
        else:
            print("No action taken. Pass --push or --output-dir to save the constructed dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpeechEditBench acoustic retrieval task.")
    parser.add_argument("--repo-id", type=str, default="deep9539/speech_edit_acoustic", help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face write token")
    parser.add_argument("--push", action="store_true", help="Push dataset to HF Hub")
    parser.add_argument("--output-dir", type=Path, help="Local directory to save processed datasets")
    args = parser.parse_args()

    process_and_upload(
        repo_id=args.repo_id,
        token=args.token,
        push=args.push,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
