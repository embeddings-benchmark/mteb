#!/usr/bin/env python3
"""
Script to prepare and upload the SSW60 (Sapsucker Woods 60) standalone audio and image 
subsets to the Hugging Face Hub under separate configurations.

The script:
1. Loads taxa and metadata for:
   - Macaulay Library standalone audios (audio_ml)
   - iNaturalist static images (images_inat)
   - NABirds static images (images_nabirds)
2. Formats and organizes them into standard Hugging Face `Image` and `Audio` features.
3. Groups them into separate dataset configurations: 'audio', 'images_inat', and 'images_nabirds'.
4. Pushes each configuration to the Hugging Face Hub (e.g., nik1995/ssw60_audio_image).

Requirements:
    pip install datasets pandas pillow tqdm huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Features, Image, Value
from tqdm import tqdm


def build_config_dataset(
    metadata_path: Path,
    media_dir: Path,
    taxa_map: dict,
    config_name: str,
    debug: bool = False,
) -> Optional[DatasetDict]:
    """
    Builds a DatasetDict for a given configuration (audio, images_inat, or images_nabirds).
    """
    if not metadata_path.exists() or not media_dir.exists():
        print(f"Warning: Missing metadata or media folder for config '{config_name}'. Skipping.")
        return None

    print(f"\nProcessing config '{config_name}' from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} total rows from {metadata_path.name}")

    if debug:
        print(f"Debug mode: Processing up to 5 samples per split for config '{config_name}'.")
        debug_dfs = []
        for split in df["split"].unique():
            debug_dfs.append(df[df["split"] == split].head(5))
        df = pd.concat(debug_dfs)

    # Determine asset_id type
    # For NABirds, asset_id is a hex string. For Others, it is integer.
    is_nabirds = (config_name == "images_nabirds")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {config_name}"):
        raw_asset_id = row["asset_id"]
        # nabirds is hex string, others are int
        asset_id = str(raw_asset_id) if is_nabirds else int(raw_asset_id)
        label = int(row["label"])
        split = str(row["split"])

        # Construct file name
        ext = "wav" if config_name == "audio" else "jpg"
        file_path = media_dir / f"{asset_id}.{ext}"

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue

        # Get taxonomic info
        species_info = taxa_map.get(label, {
            "species_code": "unknown",
            "common_name": "unknown",
            "scientific_name": "unknown",
            "family": "unknown",
            "order": "unknown",
        })

        record = {
            "asset_id": str(asset_id),
            "label": label,
            "species_code": species_info["species_code"],
            "common_name": species_info["common_name"],
            "scientific_name": species_info["scientific_name"],
            "family": species_info["family"],
            "order": species_info["order"],
            "split": split,
        }

        # Add modality-specific fields
        if config_name == "audio":
            record["audio"] = str(file_path)
            record["samplerate"] = int(row["samplerate"]) if not pd.isna(row["samplerate"]) else 0
            record["channels"] = int(row["channels"]) if not pd.isna(row["channels"]) else 0
            record["samples"] = int(row["samples"]) if not pd.isna(row["samples"]) else 0
            record["duration_seconds"] = float(row["duration_seconds"]) if not pd.isna(row["duration_seconds"]) else 0.0
        else:
            record["image"] = str(file_path)
            record["height"] = int(row["height"]) if not pd.isna(row["height"]) else 0
            record["width"] = int(row["width"]) if not pd.isna(row["width"]) else 0
            record["channels"] = int(row["channels"]) if not pd.isna(row["channels"]) else 0
            if config_name == "images_inat":
                record["rights_holder"] = str(row["rights_holder"]) if not pd.isna(row["rights_holder"]) else ""
                record["license_id"] = int(row["license_id"]) if not pd.isna(row["license_id"]) else -1
            elif config_name == "images_nabirds":
                record["photographer"] = str(row["photographer"]) if not pd.isna(row["photographer"]) else ""

        records.append(record)

    if not records:
        print(f"Warning: No valid records found for config '{config_name}'.")
        return None

    df_records = pd.DataFrame(records)

    # Define Schema features
    base_features = {
        "asset_id": Value("string"),
        "label": Value("int64"),
        "species_code": Value("string"),
        "common_name": Value("string"),
        "scientific_name": Value("string"),
        "family": Value("string"),
        "order": Value("string"),
        "split": Value("string"),
    }

    if config_name == "audio":
        schema_features = Features({
            "audio": Audio(sampling_rate=16000), # standard 16kHz sampling rate
            "samplerate": Value("int64"),
            "channels": Value("int64"),
            "samples": Value("int64"),
            "duration_seconds": Value("float64"),
            **base_features,
        })
    elif config_name == "images_inat":
        schema_features = Features({
            "image": Image(),
            "height": Value("int64"),
            "width": Value("int64"),
            "channels": Value("int64"),
            "rights_holder": Value("string"),
            "license_id": Value("int64"),
            **base_features,
        })
    elif config_name == "images_nabirds":
        schema_features = Features({
            "image": Image(),
            "height": Value("int64"),
            "width": Value("int64"),
            "channels": Value("int64"),
            "photographer": Value("string"),
            **base_features,
        })

    splits_dict = {}
    for split_name in df_records["split"].unique():
        split_df = df_records[df_records["split"] == split_name]
        split_dict = {col: split_df[col].tolist() for col in df_records.columns}
        dataset = Dataset.from_dict(split_dict, features=schema_features)
        splits_dict[split_name] = dataset
        print(f"Created subset '{config_name}' split '{split_name}' with {len(dataset)} samples.")

    return DatasetDict(splits_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload SSW60 standalone audio and image subsets to Hugging Face Hub."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the downloaded and extracted ssw60 dataset directory (e.g., ~/Downloads/ssw60)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nik1995/ssw60_audio_image",
        help="Hugging Face repository ID (default: nik1995/ssw60_audio_image)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only processes up to 5 samples per split)",
    )

    args = parser.parse_args()

    # Resolve HF Token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: Hugging Face token not provided. You can log in via huggingface-cli or provide HF_TOKEN environment variable.")

    # Expand user paths (like ~/)
    data_dir = Path(os.path.expanduser(args.data_dir))
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    taxa_csv = data_dir / "taxa.csv"
    if not taxa_csv.exists():
        print(f"Error: Missing taxa.csv in '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Load taxa metadata to build taxonomy mapping
    print("Loading taxa taxonomy metadata...")
    taxa_df = pd.read_csv(taxa_csv)
    taxa_map = {}
    for _, row in taxa_df.iterrows():
        taxa_map[int(row["label"])] = {
            "species_code": str(row["species_code"]),
            "common_name": str(row["common_name"]),
            "scientific_name": str(row["scientific_name"]),
            "family": str(row["family"]),
            "order": str(row["order"]),
        }
    print(f"Found {len(taxa_map)} species classes in taxa.csv")

    # List of configurations to process
    configs = [
        {
            "name": "audio",
            "metadata": data_dir / "audio_ml.csv",
            "media": data_dir / "audio_ml",
        },
        {
            "name": "images_inat",
            "metadata": data_dir / "images_inat.csv",
            "media": data_dir / "images_inat",
        },
        {
            "name": "images_nabirds",
            "metadata": data_dir / "images_nabirds.csv",
            "media": data_dir / "images_nabirds",
        },
    ]

    # Process and upload each configuration
    for cfg in configs:
        dataset_dict = build_config_dataset(
            metadata_path=cfg["metadata"],
            media_dir=cfg["media"],
            taxa_map=taxa_map,
            config_name=cfg["name"],
            debug=args.debug,
        )

        if dataset_dict is None:
            continue

        print(f"Pushing subset '{cfg['name']}' to Hugging Face Hub: {args.repo_id}")
        try:
            dataset_dict.push_to_hub(
                args.repo_id,
                config_name=cfg["name"],
                token=token,
                private=False,
            )
            print(f"✓ Successfully uploaded subset '{cfg['name']}' to https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"✗ Error uploading subset '{cfg['name']}': {e}", file=sys.stderr)
            print("You can try pushing the dataset manually or log in via 'huggingface-cli login'.")


if __name__ == "__main__":
    main()
