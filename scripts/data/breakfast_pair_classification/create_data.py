#!/usr/bin/env python3
"""Build and upload Breakfast video pair classification for MTEB.

Samples balanced same-class / different-class video pairs from mteb/Breakfast
with a fixed seed (same logic as scripts/upload_video_pair_classification.py).

Hub dataset:
  - {repo-id}  (default: Wissam42/Breakfast-PC-V)

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/breakfast_pair_classification/create_data.py \\
      --repo-id Wissam42/Breakfast-PC-V \\
      --push
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import create_repo

_SOURCE = "mteb/Breakfast"
_SOURCE_REVISION = "59a874899eb241993794a3454c37829727c3b559"
_SEED = 42
_MAX_PER_SIDE = 1024


def generate_pairs(
    class_labels: list,
    rng: random.Random,
    max_per_side: int = 1024,
) -> list[tuple[int, int, int]]:
    label_groups: dict[object, list[int]] = defaultdict(list)
    for i, label in enumerate(class_labels):
        label_groups[label].append(i)

    all_labels = list(label_groups.keys())
    pos_pairs: list[tuple[int, int]] = []
    neg_pairs: list[tuple[int, int]] = []
    indices = list(range(len(class_labels)))
    rng.shuffle(indices)

    for i in indices:
        cls = class_labels[i]
        same = [j for j in label_groups[cls] if j != i]
        if same and len(pos_pairs) < max_per_side:
            pos_pairs.append((i, rng.choice(same)))
        others = [label for label in all_labels if label != cls]
        if others and len(neg_pairs) < max_per_side:
            neg_cls = rng.choice(others)
            neg_pairs.append((i, rng.choice(label_groups[neg_cls])))
        if len(pos_pairs) >= max_per_side and len(neg_pairs) >= max_per_side:
            break

    n = min(len(pos_pairs), len(neg_pairs))
    pairs: list[tuple[int, int, int]] = [(a, b, 1) for a, b in pos_pairs[:n]]
    pairs += [(a, b, 0) for a, b in neg_pairs[:n]]
    rng.shuffle(pairs)
    return pairs


def build_pair_dataset(
    ds: Dataset,
    pairs: list[tuple[int, int, int]],
) -> Dataset:
    idx1 = [p[0] for p in pairs]
    idx2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    chunk_size = 64
    chunks: list[Dataset] = []
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        d1 = (
            ds.select(idx1[start:end])
            .select_columns(["video"])
            .rename_columns({"video": "video1"})
        )
        d2 = (
            ds.select(idx2[start:end])
            .select_columns(["video"])
            .rename_columns({"video": "video2"})
        )
        chunk = concatenate_datasets([d1, d2], axis=1)
        chunk = chunk.add_column("label", labels[start:end])
        chunks.append(chunk)

    return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Wissam42/Breakfast-PC-V")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/breakfast_pc"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(_SOURCE, revision=_SOURCE_REVISION, split="test")
    print(f"Loaded {len(ds)} test videos from {_SOURCE}")

    pairs = generate_pairs(ds["label"], random.Random(_SEED), _MAX_PER_SIDE)
    out = build_pair_dataset(ds, pairs)
    print(f"Generated {len(out)} pairs")

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        out.push_to_hub(args.repo_id, split="test", token=token)
        print(f"Pushed {args.repo_id}")
    else:
        out_dir = args.work_dir / "mteb_export"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.save_to_disk(out_dir)
        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
