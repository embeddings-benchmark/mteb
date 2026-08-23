"""Construct image pair-classification datasets from mteb image classification mirrors.

For each source dataset's test split, balanced pairs are sampled with a fixed
seed: half "same" pairs (two distinct images sharing a class label) and half
"different" pairs (images from two distinct classes). Labels: 1 = same class,
0 = different class. Pairs are deduplicated.

Run: python scripts/build_image_pair_classification.py [--push]
"""

from __future__ import annotations

import argparse
import random

from datasets import Dataset, DatasetDict, Image as ImageFeature, load_dataset

SEED = 42

SOURCES = [
    # (source repo, revision, image col, label col, split, n_pairs, target repo)
    ("mteb/dtd", None, "image", "label", "test", 3000, "DTDPairClassification"),
    (
        "mteb/FGVCAircraftZeroShot",
        None,
        "image",
        "variant",
        "test",
        2000,
        "FGVCAircraftPairClassification",
    ),
    (
        "mteb/oxford-flowers",
        None,
        "image",
        "label",
        "test",
        2000,
        "OxfordFlowersPairClassification",
    ),
    (
        "mteb/eurosat-rgb",
        None,
        "image",
        "label",
        "test",
        4000,
        "EuroSATPairClassification",
    ),
    ("mteb/wds_fer2013", None, "jpg", "cls", "test", 4000, "FER2013PairClassification"),
]


def build_pairs(ds, image_col, label_col, n_pairs, rng):
    by_class: dict[int, list[int]] = {}
    for i, lab in enumerate(ds[label_col]):
        by_class.setdefault(lab, []).append(i)
    classes = [c for c, idxs in by_class.items() if len(idxs) >= 2]

    seen = set()
    same, diff = [], []
    n_half = n_pairs // 2
    while len(same) < n_half:
        c = rng.choice(classes)
        a, b = rng.sample(by_class[c], 2)
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        same.append((a, b, 1))
    all_classes = list(by_class)
    while len(diff) < n_half:
        c1, c2 = rng.sample(all_classes, 2)
        a = rng.choice(by_class[c1])
        b = rng.choice(by_class[c2])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        diff.append((a, b, 0))

    pairs = same + diff
    rng.shuffle(pairs)
    images = ds[image_col]
    return (
        Dataset.from_dict(
            {
                "image1": [images[a] for a, _, _ in pairs],
                "image2": [images[b] for _, b, _ in pairs],
                "label": [l for _, _, l in pairs],
            }
        )
        .cast_column("image1", ImageFeature())
        .cast_column("image2", ImageFeature())
    )


def main(push: bool, namespace: str):
    for src, rev, image_col, label_col, split, n_pairs, target in SOURCES:
        print(f"== {src} -> {namespace}/{target}")
        ds = load_dataset(src, revision=rev, split=split)
        rng = random.Random(SEED)
        pairs = build_pairs(ds, image_col, label_col, n_pairs, rng)
        n_same = sum(pairs["label"])
        print(
            f"   {len(pairs)} pairs ({n_same} same / {len(pairs) - n_same} different)"
        )
        out = DatasetDict({"test": pairs})
        if push:
            out.push_to_hub(f"{namespace}/{target}")
            print(f"   pushed to {namespace}/{target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--namespace", default="shriyasudhakar")
    args = parser.parse_args()
    main(args.push, args.namespace)
