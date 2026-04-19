from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def generate_pairs(
    class_labels: list[int],
    rng: random.Random,
    max_per_side: int = 1024,
) -> list[tuple[int, int, int]]:
    """Generate balanced positive/negative pairs from classification labels."""
    label_groups: dict[int, list[int]] = defaultdict(list)
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
    pairs = [(a, b, 1) for a, b in pos_pairs[:n]] + [
        (a, b, 0) for a, b in neg_pairs[:n]
    ]
    rng.shuffle(pairs)
    return pairs


def build_pair_dataset(
    ds: Dataset,
    pairs: list[tuple[int, int, int]],
    video_column: str = "video",
) -> Dataset:
    """Build a pair classification dataset using index-based selection.

    Processes in chunks to avoid Arrow's 2 GB offset overflow when
    ``concatenate_datasets(axis=1)`` calls ``flatten_indices`` on large
    video datasets.
    """
    from datasets import concatenate_datasets

    if not pairs:
        msg = "No pairs generated — check that the dataset has at least 2 classes with ≥2 samples each."
        raise ValueError(msg)

    idx1 = [p[0] for p in pairs]
    idx2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    chunk_size = 64
    chunks: list[Dataset] = []
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        d1 = (
            ds.select(idx1[start:end])
            .select_columns([video_column])
            .rename_column(video_column, "video1")
        )
        d2 = (
            ds.select(idx2[start:end])
            .select_columns([video_column])
            .rename_column(video_column, "video2")
        )
        chunk = concatenate_datasets([d1, d2], axis=1)
        chunk = chunk.add_column("label", labels[start:end])
        chunks.append(chunk)

    return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]
