from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features, Value

ARCHIVES = {
    "FSDKaggle2018.audio_test.zip": "f85c8665c3fc39311ea1b748e194a81d",
    "FSDKaggle2018.audio_train.zip": "a8b47ff4b52022c178f88fa5c6f080d0",
    "FSDKaggle2018.doc.zip": "4f2d1ac88f33a62f9db3108b269ee1b7",
    "FSDKaggle2018.meta.zip": "f16828ac8be0e5285b9175a12f90d784",
}

EXPECTED_TRAIN_ROWS = 9_473
EXPECTED_VERIFIED_TRAIN_ROWS = 3_710
EXPECTED_TEST_ROWS = 1_600
EXPECTED_LABELS = 41
EXPECTED_SOURCE_LICENSES = {
    "Attribution",
    "Attribution Noncommercial",
    "Creative Commons 0",
}
ARTIFACT_MANIFEST_FILENAME = "fsdkaggle2018_artifact_manifest.json"
ZENODO_RECORD_ID = 2_552_860


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies published archive checksums
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_extract(source_dir: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in ARCHIVES.items():
        archive = source_dir / filename
        if not archive.is_file():
            raise FileNotFoundError(f"Missing source archive: {archive}")
        actual_md5 = _md5(archive)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Checksum mismatch for {archive}: {actual_md5} != {expected_md5}"
            )
        with zipfile.ZipFile(archive) as zip_file:
            extraction_root = extracted_dir.resolve()
            for member in zip_file.infolist():
                destination = (extracted_dir / member.filename).resolve()
                if not destination.is_relative_to(extraction_root):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            zip_file.extractall(extracted_dir)


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _validate_metadata_rows(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[str]]:
    train_columns = {"fname", "label", "manually_verified", "freesound_id", "license"}
    test_columns = {"fname", "label", "usage", "freesound_id", "license"}
    if not train_rows or set(train_rows[0]) != train_columns:
        raise ValueError("Unexpected FSDKaggle2018 training metadata schema")
    if not test_rows or set(test_rows[0]) != test_columns:
        raise ValueError("Unexpected FSDKaggle2018 test metadata schema")

    verified_train_rows = [row for row in train_rows if row["manually_verified"] == "1"]
    label_names = sorted({row["label"] for row in train_rows + test_rows})
    selected_rows = verified_train_rows + test_rows
    selected_filenames = [row["fname"] for row in selected_rows]

    if (
        len(train_rows) != EXPECTED_TRAIN_ROWS
        or len(verified_train_rows) != EXPECTED_VERIFIED_TRAIN_ROWS
    ):
        raise ValueError("Unexpected FSDKaggle2018 training metadata counts")
    if len(test_rows) != EXPECTED_TEST_ROWS or len(label_names) != EXPECTED_LABELS:
        raise ValueError("Unexpected FSDKaggle2018 test or label counts")
    if len(selected_filenames) != len(set(selected_filenames)):
        raise ValueError("Selected FSDKaggle2018 filenames must be unique")
    if {row["license"] for row in selected_rows} != EXPECTED_SOURCE_LICENSES:
        raise ValueError("Unexpected FSDKaggle2018 source-license values")
    if any(not row["freesound_id"].isdigit() for row in selected_rows):
        raise ValueError("FSDKaggle2018 Freesound IDs must be numeric")
    return verified_train_rows, label_names


def _build_split(
    rows: list[dict[str, str]],
    audio_dir: Path,
    label_names: list[str],
) -> Dataset:
    audio_paths = [audio_dir / row["fname"] for row in rows]
    missing_audio = [path for path in audio_paths if not path.is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Missing audio file: {missing_audio[0]}")

    features = Features(
        {
            "id": Value("string"),
            "audio": Audio(),
            "label": ClassLabel(names=label_names),
            "freesound_id": Value("string"),
            "source_url": Value("string"),
            "source_license": Value("string"),
        }
    )
    dataset = Dataset.from_dict(
        {
            "id": [row["fname"].removesuffix(".wav") for row in rows],
            "audio": [str(path) for path in audio_paths],
            "label": [label_names.index(row["label"]) for row in rows],
            "freesound_id": [row["freesound_id"] for row in rows],
            "source_url": [
                f"https://freesound.org/s/{row['freesound_id']}/" for row in rows
            ],
            "source_license": [row["license"] for row in rows],
        },
        features=features,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {"archives": ARCHIVES, "labels": label_names, "rows": rows},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return dataset.select(range(len(dataset)), new_fingerprint=fingerprint)


def _write_artifact_manifest(
    output_dir: Path,
    source_dir: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
    audio_dirs: dict[str, Path],
    label_names: list[str],
) -> None:
    files = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            sample_id = row["fname"].removesuffix(".wav")
            audio_path = audio_dirs[split] / row["fname"]
            files[sample_id] = {
                "split": split,
                "source_filename": row["fname"],
                "label": row["label"],
                "freesound_id": row["freesound_id"],
                "source_url": f"https://freesound.org/s/{row['freesound_id']}/",
                "source_license": row["license"],
                "sha256": _sha256(audio_path),
            }

    manifest = {
        "schema_version": 1,
        "zenodo_record_id": ZENODO_RECORD_ID,
        "selection": "manually verified training clips and all scored test clips",
        "archives": {
            filename: {
                "md5": expected_md5,
                "size": (source_dir / filename).stat().st_size,
            }
            for filename, expected_md5 in sorted(ARCHIVES.items())
        },
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "labels": label_names,
        "source_license_counts": {
            split: dict(sorted(Counter(row["license"] for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "files": dict(sorted(files.items())),
    }
    manifest_path = output_dir / ARTIFACT_MANIFEST_FILENAME
    temporary_path = manifest_path.with_suffix(".json.part")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(manifest_path)


def create_dataset(source_dir: Path, work_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )

    extracted_dir = work_dir / "extracted"
    _validate_and_extract(source_dir, extracted_dir)

    metadata_dir = extracted_dir / "FSDKaggle2018.meta"
    train_rows = _read_metadata(metadata_dir / "train_post_competition.csv")
    test_rows = _read_metadata(metadata_dir / "test_post_competition_scoring_clips.csv")
    verified_train_rows, label_names = _validate_metadata_rows(train_rows, test_rows)

    audio_dirs = {
        "train": extracted_dir / "FSDKaggle2018.audio_train",
        "test": extracted_dir / "FSDKaggle2018.audio_test",
    }
    rows_by_split = {"train": verified_train_rows, "test": test_rows}
    dataset = DatasetDict(
        {
            "train": _build_split(
                verified_train_rows,
                audio_dirs["train"],
                label_names,
            ),
            "test": _build_split(
                test_rows,
                audio_dirs["test"],
                label_names,
            ),
        }
    )
    dataset.save_to_disk(output_dir)
    _write_artifact_manifest(
        output_dir,
        source_dir,
        rows_by_split,
        audio_dirs,
        label_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the manually verified FSDKaggle2018 MTEB dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the four official Zenodo zip archives.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Directory used for checksum-verified archive extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory where the DatasetDict is saved.",
    )
    args = parser.parse_args()
    create_dataset(args.source_dir, args.work_dir, args.output_dir)


if __name__ == "__main__":
    main()
