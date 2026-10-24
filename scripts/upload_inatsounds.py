#!/usr/bin/env python3
"""
Script to download iNatSounds 2024 test set and upload to HuggingFace Hub.

Usage:
    python scripts/upload_inatsounds.py --token YOUR_HF_TOKEN

    Or set HF_TOKEN environment variable:
    export HF_TOKEN=YOUR_HF_TOKEN
    python scripts/upload_inatsounds.py

Requirements:
    pip install datasets huggingface_hub tqdm requests
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List

import requests
from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi
from tqdm import tqdm


# Download URLs from AWS Open Data Program
TEST_RECORDINGS_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/test.tar.gz"
)
TEST_ANNOTATIONS_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/test.json.tar.gz"
)

# MD5 checksums for verification
TEST_RECORDINGS_MD5 = "082a2e96ca34f6c9f61767e33cbf3626"
TEST_ANNOTATIONS_MD5 = "5482f112f05db4ba8576ba425d95d6a7"


def download_file(url: str, output_path: Path, expected_md5: str = None) -> None:
    """Download a file with progress bar and optional MD5 verification."""
    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    md5_hash = hashlib.md5() if expected_md5 else None

    with (
        open(output_path, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
                if md5_hash:
                    md5_hash.update(chunk)

    if expected_md5:
        calculated_md5 = md5_hash.hexdigest()
        if calculated_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch! Expected {expected_md5}, got {calculated_md5}"
            )
        print(f"✓ MD5 checksum verified")


def extract_tar_gz(tar_path: Path, extract_to: Path) -> None:
    """Extract a tar.gz file with progress bar."""
    print(f"Extracting {tar_path.name}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_to)
                pbar.update(1)


def parse_coco_annotations(annotations_path: Path) -> Dict:
    """Parse COCO-style JSON annotations."""
    print(f"Parsing annotations from {annotations_path}...")

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Create category ID to name mapping
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Create audio ID to file name mapping
    audio_files = {audio["id"]: audio["file_name"] for audio in data["audio"]}

    # Create mapping from audio ID to category name
    audio_to_category = {}
    for ann in data["annotations"]:
        audio_id = ann["audio_id"]
        category_id = ann["category_id"]
        audio_to_category[audio_id] = categories[category_id]

    return {
        "audio_files": audio_files,
        "audio_to_category": audio_to_category,
        "categories": categories,
    }


def create_dataset(audio_dir: Path, annotations_info: Dict) -> Dataset:
    """Create HuggingFace Dataset from audio files and annotations."""
    print("Creating HuggingFace Dataset...")

    audio_files = annotations_info["audio_files"]
    audio_to_category = annotations_info["audio_to_category"]

    # Collect all audio files and their labels
    data = []
    missing_files = []

    for audio_id, file_name in tqdm(audio_files.items(), desc="Processing audio files"):
        audio_path = audio_dir / file_name

        if not audio_path.exists():
            missing_files.append(str(audio_path))
            continue

        label = audio_to_category.get(audio_id)
        if label is None:
            print(f"Warning: No label found for audio_id {audio_id}")
            continue

        data.append(
            {
                "audio": str(audio_path),
                "label": label,
            }
        )

    if missing_files:
        print(f"Warning: {len(missing_files)} audio files not found")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"  Missing: {f}")

    print(f"Creating dataset with {len(data)} samples...")

    # Define features
    features = Features(
        {
            "audio": Audio(sampling_rate=22050),
            "label": Value("string"),
        }
    )

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "audio": [item["audio"] for item in data],
            "label": [item["label"] for item in data],
        },
        features=features,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download iNatSounds 2024 test set and upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mteb/inat_sounds",
        help="HuggingFace repository ID (default: mteb/inat_sounds)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for downloads (default: system temp)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after upload",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Provide via --token or HF_TOKEN env variable"
        )

    # Create temporary directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="inatsounds_"))
        cleanup_temp = not args.keep_temp

    try:
        print(f"Working directory: {temp_dir}")

        # Download annotations
        annotations_tar = temp_dir / "test.json.tar.gz"
        annotations_json = temp_dir / "test.json"
        if not annotations_json.exists():
            download_file(
                TEST_ANNOTATIONS_URL,
                annotations_tar,
                expected_md5=TEST_ANNOTATIONS_MD5,
            )
            extract_tar_gz(annotations_tar, temp_dir)

        # Parse annotations
        annotations_info = parse_coco_annotations(annotations_json)
        print(f"Found {len(annotations_info['categories'])} categories")
        print(f"Found {len(annotations_info['audio_files'])} audio files")

        # Download recordings
        recordings_tar = temp_dir / "test.tar.gz"
        audio_dir = temp_dir
        test_dir = temp_dir / "test"
        if not test_dir.exists() or len(list(test_dir.iterdir())) == 0:
            download_file(
                TEST_RECORDINGS_URL,
                recordings_tar,
                expected_md5=TEST_RECORDINGS_MD5,
            )
            extract_tar_gz(recordings_tar, temp_dir)
        else:
            print(f"Audio files already extracted in {test_dir}, skipping download")

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Create dataset
        dataset = create_dataset(audio_dir, annotations_info)

        print(f"\nDataset info:")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Features: {dataset.features}")

        # Create DatasetDict with only test split
        dataset_dict = DatasetDict({"test": dataset})

        # Push to HuggingFace Hub
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
        dataset_dict.push_to_hub(
            args.repo_id,
            token=token,
            private=False,
        )

        print(
            f"\n✓ Successfully uploaded to https://huggingface.co/datasets/{args.repo_id}"
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

    finally:
        # Cleanup temporary directory if needed
        if cleanup_temp and temp_dir.exists():
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        elif not cleanup_temp:
            print(f"\nTemporary files kept in: {temp_dir}")


if __name__ == "__main__":
    main()
