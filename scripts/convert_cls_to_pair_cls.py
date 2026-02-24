"""
General-purpose script to convert any classification dataset into
PairClassification format for MTEB.

For each sample, creates positive pairs (same label) and negative pairs
(different label). Automatically detects and preserves video/audio/image/text
columns, renaming them with "1"/"2" suffixes for the pair.

Examples:
    # Dry run — just prints stats
    python scripts/convert_cls_to_pair_cls.py \
        --source mteb/Human-Animal-Cartoon \
        --label_col source \
        --input_cols video audio

    # Push to Hub
    python scripts/convert_cls_to_pair_cls.py \
        --source mteb/Human-Animal-Cartoon \
        --label_col source \
        --input_cols video audio \
        --target mteb/Human-Animal-Cartoon-PairClassification \
        --push_to_hub

    # RAVDESS audio-only
    python scripts/convert_cls_to_pair_cls.py \
        --source mteb/RAVDESS_AV \
        --label_col emotion \
        --input_cols video audio \
        --target mteb/RAVDESS-AV-PairClassification \
        --push_to_hub

    # Save locally
    python scripts/convert_cls_to_pair_cls.py \
        --source mteb/Human-Animal-Cartoon \
        --label_col source \
        --input_cols video audio \
        --save_dir ./output_pair_cls
"""

import argparse
import random
from collections import defaultdict

from datasets import Audio, Dataset, Video, load_dataset
from tqdm import tqdm


def create_pairs(
    dataset: Dataset,
    label_column: str,
    input_columns: list[str],
    n_positive_per_sample: int = 3,
    n_negative_per_sample: int = 3,
    seed: int = 42,
) -> dict[str, list]:
    """Create positive and negative pairs from a classification dataset.

    Args:
        dataset: Source HuggingFace Dataset.
        label_column: Column name containing the class labels.
        input_columns: List of columns to include in the pairs (e.g. ["video", "audio"]).
        n_positive_per_sample: Max positive pairs to create per sample.
        n_negative_per_sample: Max negative pairs to create per sample.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys like "video1", "audio1", "video2", "audio2", "label".
    """
    random.seed(seed)

    # Group indices by class
    class_to_indices: dict = defaultdict(list)
    for i, label in enumerate(dataset[label_column]):
        class_to_indices[label].append(i)

    all_classes = list(class_to_indices.keys())
    print(f"Found {len(all_classes)} classes: {all_classes}")
    for cls, indices in class_to_indices.items():
        print(f"  {cls}: {len(indices)} samples")

    # Initialize pair columns
    pairs: dict[str, list] = {}
    for col in input_columns:
        pairs[f"{col}1"] = []
        pairs[f"{col}2"] = []
    pairs["label"] = []

    seen_pairs: set[tuple[int, int]] = set()

    def add_pair(i: int, j: int, label: int):
        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        for col in input_columns:
            pairs[f"{col}1"].append(dataset[i][col])
            pairs[f"{col}2"].append(dataset[j][col])
        pairs["label"].append(label)

    for i in tqdm(range(len(dataset)), desc="Creating pairs"):
        sample_class = dataset[i][label_column]

        # Positive pairs: same class
        same_class_indices = [idx for idx in class_to_indices[sample_class] if idx != i]
        n_pos = min(n_positive_per_sample, len(same_class_indices))
        for j in random.sample(same_class_indices, n_pos):
            add_pair(i, j, label=1)

        # Negative pairs: different class
        other_classes = [c for c in all_classes if c != sample_class]
        neg_candidates = []
        for c in other_classes:
            neg_candidates.extend(class_to_indices[c])
        n_neg = min(n_negative_per_sample, len(neg_candidates))
        for j in random.sample(neg_candidates, n_neg):
            add_pair(i, j, label=0)

    return pairs


def cast_feature_types(
    pair_ds: Dataset, source_ds: Dataset, input_columns: list[str]
) -> Dataset:
    """Re-apply feature types (Video, Audio) from the source to the paired dataset."""
    for col in input_columns:
        feat = source_ds.features.get(col)
        if isinstance(feat, Video):
            pair_ds = pair_ds.cast_column(f"{col}1", Video())
            pair_ds = pair_ds.cast_column(f"{col}2", Video())
        elif isinstance(feat, Audio):
            sr = feat.sampling_rate
            pair_ds = pair_ds.cast_column(f"{col}1", Audio(sampling_rate=sr))
            pair_ds = pair_ds.cast_column(f"{col}2", Audio(sampling_rate=sr))
    return pair_ds


def main():
    parser = argparse.ArgumentParser(
        description="Convert any classification dataset to PairClassification format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source dataset path on HuggingFace Hub (e.g. mteb/Human-Animal-Cartoon)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to process (default: test)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        required=True,
        help="Column name containing class labels (e.g. source, emotion, action)",
    )
    parser.add_argument(
        "--input_cols",
        type=str,
        nargs="+",
        required=True,
        help="Column names to include in pairs (e.g. video audio text)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target dataset name for HuggingFace Hub",
    )
    parser.add_argument(
        "--n_positive",
        type=int,
        default=3,
        help="Max positive pairs per sample (default: 3)",
    )
    parser.add_argument(
        "--n_negative",
        type=int,
        default=3,
        help="Max negative pairs per sample (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the resulting dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Local directory to save the dataset",
    )
    args = parser.parse_args()

    # Validate
    if args.push_to_hub and not args.target:
        parser.error("--target is required when using --push_to_hub")

    # Load
    print(f"Loading: {args.source} (split={args.split})")
    ds = load_dataset(args.source, split=args.split)
    print(f"  {len(ds)} samples, columns: {ds.column_names}")

    # Validate columns exist
    missing = [
        c for c in [args.label_col] + args.input_cols if c not in ds.column_names
    ]
    if missing:
        parser.error(
            f"Columns not found in dataset: {missing}. Available: {ds.column_names}"
        )

    # Create pairs
    print(f"\nCreating pairs (pos={args.n_positive}, neg={args.n_negative})...")
    pairs = create_pairs(
        ds,
        label_column=args.label_col,
        input_columns=args.input_cols,
        n_positive_per_sample=args.n_positive,
        n_negative_per_sample=args.n_negative,
        seed=args.seed,
    )

    n_pos = sum(1 for l in pairs["label"] if l == 1)
    n_neg = sum(1 for l in pairs["label"] if l == 0)
    print(f"\nResult: {len(pairs['label'])} pairs ({n_pos} positive, {n_neg} negative)")

    pair_ds = Dataset.from_dict(pairs)
    pair_ds = cast_feature_types(pair_ds, ds, args.input_cols)

    print(f"Columns: {pair_ds.column_names}")
    print(f"Features: {pair_ds.features}")

    # Output
    if args.save_dir:
        print(f"\nSaving to {args.save_dir}...")
        pair_ds.save_to_disk(args.save_dir)

    if args.push_to_hub:
        print(f"\nPushing to Hub as {args.target}...")
        pair_ds.push_to_hub(args.target, split=args.split)

    if not args.save_dir and not args.push_to_hub:
        print("\nDry run complete. Use --push_to_hub or --save_dir to persist.")


if __name__ == "__main__":
    main()
