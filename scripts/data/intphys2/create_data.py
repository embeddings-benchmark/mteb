"""Build the MTEB IntPhys2 dataset from the official labeled Main split.

Example:
    uv run python scripts/data/intphys2/create_data.py \
        --download --source-dir /path/to/source --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, Video
from huggingface_hub import snapshot_download

SOURCE_REPO = "facebook/IntPhys2"
SOURCE_REVISION = "a077a2f94e25889016fc6e5983cf21e2ddc25fb2"
SOURCE_METADATA_SHA256 = (
    "0db123911e595e263e88dde19ac52bc2684d23d19d7c1f2dc49c67c916075422"
)
EXPECTED_ROWS = 1_012
EXPECTED_SCENES = 253
EXPECTED_VARIANTS = {
    "1_Possible",
    "1_Impossible",
    "2_Possible",
    "2_Impossible",
}
TYPE_TO_LABEL = {
    "1_Impossible": 0,
    "2_Impossible": 0,
    "1_Possible": 1,
    "2_Possible": 1,
}
LABEL_NAMES = [
    "object behavior is inconsistent with Earth's physical laws",
    "object behavior is consistent with Earth's physical laws",
]
CANDIDATE_LABELS = [f"a video where {label}" for label in LABEL_NAMES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source_dir: Path) -> None:
    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        allow_patterns=["Main/**", "README.md"],
        local_dir=source_dir,
    )


def load_source(source_dir: Path) -> tuple[list[Path], list[int]]:
    main_dir = source_dir / "Main"
    metadata_path = main_dir / "metadata.csv"
    if sha256(metadata_path) != SOURCE_METADATA_SHA256:
        raise ValueError("IntPhys2 metadata does not match the pinned source revision")

    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    required_columns = {"SceneIndex", "name", "file_name", "type"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("Unexpected IntPhys2 metadata schema")

    sample_ids: set[str] = set()
    variants_by_scene: dict[str, set[str]] = {}
    video_paths: list[Path] = []
    labels: list[int] = []
    for row in rows:
        sample_id = row["name"]
        if sample_id in sample_ids or not re.fullmatch(r"[0-9a-f]{64}", sample_id):
            raise ValueError(f"Invalid or duplicate sample id: {sample_id}")
        sample_ids.add(sample_id)

        variant = row["type"]
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Unexpected variant: {variant}")
        variants_by_scene.setdefault(row["SceneIndex"], set()).add(variant)

        relative_path = Path(row["file_name"])
        if relative_path.parts != ("Videos", f"{sample_id}.mp4"):
            raise ValueError(f"Unexpected video path: {relative_path}")
        video_path = main_dir / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        video_paths.append(video_path)
        labels.append(TYPE_TO_LABEL[variant])

    if len(variants_by_scene) != EXPECTED_SCENES or any(
        variants != EXPECTED_VARIANTS for variants in variants_by_scene.values()
    ):
        raise ValueError("Unexpected scene/variant structure")
    if Counter(labels) != {0: 506, 1: 506}:
        raise ValueError(f"Unexpected label counts: {Counter(labels)}")
    return video_paths, labels


def build_dataset(source_dir: Path) -> tuple[DatasetDict, Dataset]:
    video_paths, labels = load_source(source_dir)
    default = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "video": [str(path) for path in video_paths],
                    "label": labels,
                },
                features=Features(
                    {
                        "video": Video(),
                        "label": ClassLabel(names=LABEL_NAMES),
                    }
                ),
            )
        }
    )
    candidates = Dataset.from_dict(
        {"labels": CANDIDATE_LABELS},
        features=Features({"labels": Value("string")}),
    )
    return default, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    if args.push and not args.repo_id:
        parser.error("--repo-id is required with --push")

    if args.download:
        download_source(args.source_dir)
    default, candidates = build_dataset(args.source_dir)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    default.save_to_disk(args.output_dir / "default")
    DatasetDict({"train": candidates}).save_to_disk(args.output_dir / "labels")

    if args.push:
        default.push_to_hub(args.repo_id)
        candidates.push_to_hub(args.repo_id, config_name="labels", split="train")

    counts = Counter(default["test"]["label"])
    print(
        f"Built {len(default['test'])} videos "
        f"({counts[0]} impossible, {counts[1]} possible) and "
        f"{len(candidates)} candidate labels."
    )


if __name__ == "__main__":
    main()
