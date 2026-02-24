"""
Convert the mteb/Human-Animal-Cartoon dataset from classification format
to PairClassification format.

Source format:
    video_id | video | audio | action | source
    (one sample per row, 644 rows)

Target format:
    video1 | audio1 | video2 | audio2 | label
    label = 1 if same source (human/animal/cartoon), 0 if different

Usage:
    python scripts/convert_hac_to_pair_cls.py --push_to_hub
    python scripts/convert_hac_to_pair_cls.py --save_dir ./hac_pair_cls
"""

import argparse
import random
from collections import defaultdict

from datasets import Audio, Dataset, Video, load_dataset
from tqdm import tqdm


def create_pairs(
    dataset: Dataset,
    label_column: str = "source",
    n_positive_per_sample: int = 3,
    n_negative_per_sample: int = 3,
    seed: int = 42,
) -> dict[str, list]:
    """Create positive and negative pairs from a classification dataset.

    For each sample, we create:
    - n_positive_per_sample pairs with other samples of the SAME class (label=1)
    - n_negative_per_sample pairs with samples of DIFFERENT classes (label=0)

    This produces a balanced dataset of pairs.
    """
    random.seed(seed)

    # Group indices by class
    class_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(dataset[label_column]):
        class_to_indices[label].append(i)

    all_classes = list(class_to_indices.keys())
    print(f"Found {len(all_classes)} classes: {all_classes}")
    for cls, indices in class_to_indices.items():
        print(f"  {cls}: {len(indices)} samples")

    pairs = {
        "video1": [],
        "audio1": [],
        "video2": [],
        "audio2": [],
        "label": [],
    }

    seen_pairs: set[tuple[int, int]] = set()

    for i in tqdm(range(len(dataset)), desc="Creating pairs"):
        sample_class = dataset[i][label_column]

        # Positive pairs: same class
        same_class_indices = [idx for idx in class_to_indices[sample_class] if idx != i]
        n_pos = min(n_positive_per_sample, len(same_class_indices))
        pos_partners = random.sample(same_class_indices, n_pos)

        for j in pos_partners:
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pairs["video1"].append(dataset[i]["video"])
            pairs["audio1"].append(dataset[i]["audio"])
            pairs["video2"].append(dataset[j]["video"])
            pairs["audio2"].append(dataset[j]["audio"])
            pairs["label"].append(1)

        # Negative pairs: different class
        other_classes = [c for c in all_classes if c != sample_class]
        neg_candidates = []
        for c in other_classes:
            neg_candidates.extend(class_to_indices[c])

        n_neg = min(n_negative_per_sample, len(neg_candidates))
        neg_partners = random.sample(neg_candidates, n_neg)

        for j in neg_partners:
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pairs["video1"].append(dataset[i]["video"])
            pairs["audio1"].append(dataset[i]["audio"])
            pairs["video2"].append(dataset[j]["video"])
            pairs["audio2"].append(dataset[j]["audio"])
            pairs["label"].append(0)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Convert Human-Animal-Cartoon to PairClassification format"
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="mteb/Human-Animal-Cartoon",
        help="Source dataset path on HuggingFace Hub",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="mteb/Human-Animal-Cartoon-PairClassification",
        help="Target dataset name for HuggingFace Hub",
    )
    parser.add_argument(
        "--n_positive",
        type=int,
        default=3,
        help="Number of positive pairs per sample",
    )
    parser.add_argument(
        "--n_negative",
        type=int,
        default=3,
        help="Number of negative pairs per sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
        help="Local directory to save the dataset (optional)",
    )
    args = parser.parse_args()

    print(f"Loading source dataset: {args.source_dataset}")
    ds = load_dataset(args.source_dataset, split="test")

    print(f"\nSource dataset: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"Features: {ds.features}")

    print(f"\nCreating pairs (pos={args.n_positive}, neg={args.n_negative})...")
    pairs = create_pairs(
        ds,
        label_column="source",
        n_positive_per_sample=args.n_positive,
        n_negative_per_sample=args.n_negative,
        seed=args.seed,
    )

    n_pos = sum(1 for l in pairs["label"] if l == 1)
    n_neg = sum(1 for l in pairs["label"] if l == 0)
    print(f"\nCreated {len(pairs['label'])} pairs: {n_pos} positive, {n_neg} negative")

    pair_ds = Dataset.from_dict(pairs)

    # Cast columns to proper feature types to match the source
    if "video" in ds.features and isinstance(ds.features["video"], Video):
        pair_ds = pair_ds.cast_column("video1", Video())
        pair_ds = pair_ds.cast_column("video2", Video())
    if "audio" in ds.features and isinstance(ds.features["audio"], Audio):
        sr = ds.features["audio"].sampling_rate
        pair_ds = pair_ds.cast_column("audio1", Audio(sampling_rate=sr))
        pair_ds = pair_ds.cast_column("audio2", Audio(sampling_rate=sr))

    print(f"\nPair dataset: {len(pair_ds)} rows")
    print(f"Columns: {pair_ds.column_names}")
    print(f"Features: {pair_ds.features}")
    print(f"Sample: {pair_ds[0]}")

    if args.save_dir:
        print(f"\nSaving to {args.save_dir}...")
        pair_ds.save_to_disk(args.save_dir)
        print("Done!")

    if args.push_to_hub:
        print(f"\nPushing to Hub as {args.target_dataset}...")
        pair_ds.push_to_hub(args.target_dataset, split="test")
        print("Done!")

    if not args.save_dir and not args.push_to_hub:
        print("\nNo --save_dir or --push_to_hub specified. Dry run complete.")
        print("Use --push_to_hub or --save_dir to persist the dataset.")


if __name__ == "__main__":
    main()
