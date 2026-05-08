"""Shared helpers for building VideoPairClassification datasets from
classification-style sources.

Each helper takes a HuggingFace ``Dataset`` whose rows contain (video[, audio],
class_label) and produces a balanced same-class / different-class pair dataset
suitable for :class:`mteb.abstasks.AbsTaskPairClassification`.

These helpers are invoked offline by ``scripts/upload_video_pair_classification.py``
to bake pre-paired parquet datasets that are then uploaded to HF. The task
classes load those baked datasets directly and do not call these helpers at
evaluation time (mirroring ``zachz/VideoCon-PC``, ``zachz/Vinoground-PC`` and
``zachz/AV-SpeakerBench-PC``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    from datasets import Dataset


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
        pairs: Output of :func:`generate_pairs`.
        columns: Columns to copy into ``c1`` / ``c2`` (e.g. ``("video",)`` for
            video-only or ``("video", "audio")`` for video+audio).

    Returns:
        A new ``Dataset`` with paired columns and an integer ``label`` column.
    """
    from datasets import concatenate_datasets

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
