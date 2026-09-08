"""Publish the two official HEAR Beehive States folds in MTEB format.

Download both Beehive archives from https://zenodo.org/records/6332517. Each
fold expands to roughly 31 GB, so point temporary files at a volume with enough
free space and run with the frozen audio dependencies:

    TMPDIR=/path/to/large-volume/tmp uv run --frozen --extra video \
        python scripts/data/beehive/create_data.py ARCHIVE_DIR \
        --repo-id artist/BeehiveStatesClassification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features

SAMPLE_RATE = 48_000
LABELS = ClassLabel(names=["NOQUEEN", "QUEEN"])
FEATURES = Features({"audio": Audio(sampling_rate=SAMPLE_RATE), "label": LABELS})
SPLITS = {
    "train": ("train", 256),
    "valid": ("validation", 32),
    "test": ("test", 288),
}
ARCHIVES = {
    "fold0": (
        "hear2021-beehive_states_fold0-v2-full-48000.tar.gz",
        "f9e045f9b2ddf5643edc1143304c80aa",
    ),
    "fold1": (
        "hear2021-beehive_states_fold1-v2-full-48000.tar.gz",
        "8bced7ce8336bf23fd61c61f04aee109",
    ),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive_path: Path, output_root: Path, fold: str) -> None:
    prefix = f"tasks/beehive_states_{fold}-v2-full/"
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not member.name.startswith(prefix)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unexpected archive member: {member.name}")
            archive.extract(member, output_root, filter="data")


def _publish_fold(
    archive_path: Path,
    fold: str,
    repo_id: str | None,
    output_root: Path | None,
) -> None:
    expected_md5 = ARCHIVES[fold][1]
    if _md5(archive_path) != expected_md5:
        raise ValueError(f"Checksum mismatch for {archive_path.name}")

    with tempfile.TemporaryDirectory(prefix=f"beehive-{fold}-") as directory:
        root = Path(directory)
        _extract(archive_path, root, fold)

        task_root = root / "tasks" / f"beehive_states_{fold}-v2-full"
        datasets = {}
        all_filenames = []
        for source_split, (target_split, expected_count) in SPLITS.items():
            labels = json.loads((task_root / f"{source_split}.json").read_text())
            filenames = sorted(labels)
            if len(filenames) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} examples in {fold}/{source_split}"
                )

            audio_root = task_root / str(SAMPLE_RATE) / source_split
            label_names = [labels[name][0] for name in filenames]
            if set(label_names) != set(LABELS.names):
                raise ValueError(f"Both labels must occur in {fold}/{source_split}")
            paths = [audio_root / name for name in filenames]
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError(f"Missing audio in {fold}/{source_split}")

            datasets[target_split] = Dataset.from_dict(
                {
                    "audio": [str(path) for path in paths],
                    "label": [LABELS.str2int(name) for name in label_names],
                },
                features=FEATURES,
            )
            all_filenames.extend(filenames)

        if len(all_filenames) != 576 or len(set(all_filenames)) != 576:
            raise ValueError(f"Expected 576 unique recordings in {fold}")
        dataset = DatasetDict(datasets)
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_root / fold)
        if repo_id is not None:
            dataset.push_to_hub(repo_id, config_name=fold, max_shard_size="500MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fold", choices=ARCHIVES, action="append")
    args = parser.parse_args()
    if args.repo_id is None and args.output_root is None:
        parser.error("provide --repo-id and/or --output-root")

    selected = args.fold or ARCHIVES
    for fold in selected:
        filename = ARCHIVES[fold][0]
        _publish_fold(
            args.archive_root / filename,
            fold,
            args.repo_id,
            args.output_root,
        )


if __name__ == "__main__":
    main()
