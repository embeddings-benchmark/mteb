"""Build & upload pre-baked VideoPairClassification datasets.

For each of the five MVEB classification-source datasets
(Human-Animal-Cartoon, AVE, MELD, MUSIC-AVQA, RAVDESS_AV) this script:

1. Loads the source HF dataset.
2. Generates a balanced same-class / different-class index pairing
   (using :mod:`mteb.tasks.pair_classification.eng._video_pair_helpers`).
3. Materialises ``video1``/``video2`` (and optionally ``audio1``/``audio2``)
   columns plus an integer ``label`` column.
4. Uploads the result as parquet shards to a target HF dataset repo
   (default: ``mteb/<NAME>-PC``).

The shipped task classes load these baked repos directly (mirroring how
``zachz/VideoCon-PC``, ``zachz/Vinoground-PC`` and
``zachz/AV-SpeakerBench-PC`` are wired today) and do not perform any
``dataset_transform`` at evaluation time.

Usage:
    python scripts/upload_video_pair_classification.py \
        --dataset all --owner mteb --token $HF_TOKEN
    python scripts/upload_video_pair_classification.py \
        --dataset meld --variant va

The script never deletes existing repos; it only uploads parquet shards.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from mteb.tasks.pair_classification.eng._video_pair_helpers import (
    build_pair_dataset,
    generate_pairs,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: str
    revision: str
    label_col: str
    target_suffix: str  # e.g. "Human-Animal-Cartoon-PC"


SOURCES: dict[str, SourceSpec] = {
    "hac": SourceSpec(
        "Human-Animal-Cartoon",
        "mteb/Human-Animal-Cartoon",
        "d38566c4bb055c7325314d3e46610792c2799c4b",
        "action",
        "Human-Animal-Cartoon-PC",
    ),
    "ave": SourceSpec(
        "AVE",
        "mteb/AVE-Dataset",
        "f6eb93b4e89456277a242583b5565b801bc1981d",
        "label",
        "AVE-Dataset-PC",
    ),
    "meld": SourceSpec(
        "MELD",
        "mteb/MELD",
        "6c0bf58845b1acccefc450b131c304378c1e38d5",
        "emotion",
        "MELD-PC",
    ),
    "music_avqa": SourceSpec(
        "MUSIC-AVQA",
        "mteb/MUSIC-AVQA_cls-preprocessed",
        "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        "label",
        "MUSIC-AVQA-PC",
    ),
    "ravdess": SourceSpec(
        "RAVDESS_AV",
        "mteb/RAVDESS_AV",
        "13af08387c3ce5e86c179a3718eb158669268d65",
        "emotion",
        "RAVDESS-AV-PC",
    ),
}


def push_video_dataset(
    api: HfApi, ds: Dataset, repo_id: str, split: str = "test"
) -> None:
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        approx_shards = max(1, len(ds) // 256)
        shard_size = max(1, len(ds) // approx_shards)
        total_shards = (len(ds) + shard_size - 1) // shard_size
        for shard_idx, start in enumerate(range(0, len(ds), shard_size)):
            end = min(start + shard_size, len(ds))
            fname = f"{split}-{shard_idx:05d}-of-{total_shards:05d}.parquet"
            path = os.path.join(tmpdir, fname)
            ds.select(range(start, end)).to_parquet(path)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"data/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {fname} ({end - start} rows)")


def build_pc(spec: SourceSpec, variant: str, seed: int = 42) -> Dataset:
    print(f"=== {spec.name} ({variant}) ===")
    ds = load_dataset(spec.path, revision=spec.revision, split="test")
    print(f"  Loaded {len(ds)} rows, columns={ds.column_names}")
    rng = random.Random(seed)
    pairs = generate_pairs(ds[spec.label_col], rng)
    cols = ("video", "audio") if variant == "va" else ("video",)
    out = build_pair_dataset(ds, pairs, columns=cols)
    print(f"  Generated {len(out)} pairs, columns={out.column_names}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=[*SOURCES.keys(), "all"],
        default="all",
    )
    p.add_argument(
        "--variant",
        choices=["v", "va", "both"],
        default="both",
        help="Build video-only (v), video+audio (va), or both.",
    )
    p.add_argument(
        "--owner",
        default="mteb",
        help="HF user/org that will own the new -PC repos.",
    )
    p.add_argument(
        "--token", default=None, help="HF access token (or use HF_TOKEN env)."
    )
    p.add_argument("--dry-run", action="store_true", help="Build but do not upload.")
    args = p.parse_args()

    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))

    keys = list(SOURCES) if args.dataset == "all" else [args.dataset]
    variants = ["v", "va"] if args.variant == "both" else [args.variant]

    for key in keys:
        spec = SOURCES[key]
        for variant in variants:
            out = build_pc(spec, variant)
            repo_id = f"{args.owner}/{spec.target_suffix}-{variant.upper()}"
            if args.dry_run:
                print(f"  [dry-run] would upload to {repo_id}")
                continue
            push_video_dataset(api, out, repo_id)
            print(f"  → {repo_id}")


if __name__ == "__main__":
    main()
