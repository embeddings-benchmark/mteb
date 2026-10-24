"""Build and upload pre-baked VideoPairClassification datasets.

For each of the five MVEB classification-source datasets
(Human-Animal-Cartoon, AVE, MELD, MUSIC-AVQA, RAVDESS_AV) this script:

1. Loads the source HF dataset.
2. Generates a balanced same-class / different-class index pairing.
3. Materialises ``video1``/``video2`` (and optionally ``audio1``/``audio2``)
   columns plus an integer ``label`` column.
4. Uploads the resulting dataset to a target HF repo via
   ``datasets.Dataset.push_to_hub``.

The shipped task classes load the resulting baked repos directly.

Usage:
    python scripts/upload_video_pair_classification.py \
        --dataset all --owner mteb --token $HF_TOKEN
    python scripts/upload_video_pair_classification.py \
        --dataset meld --variant va
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass

from datasets import Dataset, concatenate_datasets, load_dataset


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


def generate_pairs(
    class_labels: list,
    rng: random.Random,
    max_per_side: int = 1024,
) -> list[tuple[int, int, int]]:
    """Generate balanced positive (same-class) and negative (different-class) pairs.

    Args:
        class_labels: Per-row class labels from the source classification dataset.
        rng: Seeded ``random.Random`` instance for reproducibility.
        max_per_side: Cap on the number of positive (and negative) pairs produced.

    Returns:
        A list of ``(idx_a, idx_b, label)`` triples where ``label`` is 1 for a
        same-class pair and 0 for a different-class pair. Counts of positive and
        negative pairs are equalised.
    """
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
        others = [l for l in all_labels if l != cls]
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
    columns: tuple[str, ...] = ("video",),
) -> Dataset:
    """Materialise a paired dataset by selecting rows by index.

    For each modality column ``c`` in ``columns`` two new columns are produced:
    ``c1`` (left of pair) and ``c2`` (right of pair). A ``label`` column is
    appended.

    Args:
        ds: Source dataset.
        pairs: Output of ``generate_pairs``.
        columns: Columns to copy into ``c1`` / ``c2``.

    Returns:
        A new ``Dataset`` with paired columns and an integer ``label`` column.
    """
    if not pairs:
        raise ValueError("No pairs were generated from the source dataset")

    idx1 = [p[0] for p in pairs]
    idx2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    cols = list(columns)
    rename1 = {c: f"{c}1" for c in cols}
    rename2 = {c: f"{c}2" for c in cols}

    chunk_size = 64
    chunks: list[Dataset] = []
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        d1 = ds.select(idx1[start:end]).select_columns(cols).rename_columns(rename1)
        d2 = ds.select(idx2[start:end]).select_columns(cols).rename_columns(rename2)
        chunk = concatenate_datasets([d1, d2], axis=1)
        chunk = chunk.add_column("label", labels[start:end])
        chunks.append(chunk)

    return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]


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
    p.add_argument("--dataset", choices=[*SOURCES.keys(), "all"], default="all")
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

    token = args.token or os.environ.get("HF_TOKEN")

    keys = list(SOURCES) if args.dataset == "all" else [args.dataset]
    variants = ["v", "va"] if args.variant == "both" else [args.variant]

    for key in keys:
        spec = SOURCES[key]
        for variant in variants:
            out = build_pc(spec, variant)
            repo_id = f"{args.owner}/{spec.target_suffix}-{variant.upper()}"
            if args.dry_run:
                print(f"  [dry-run] would push to {repo_id}")
                continue
            out.push_to_hub(repo_id, split="test", token=token)
            print(f"  -> {repo_id}")


if __name__ == "__main__":
    main()
