#!/usr/bin/env python3
"""Create a strictly decodable Memotion dataset for MTEB.

The pinned source contains one truncated PNG. Its IDAT payload is incomplete,
and Pillow's permissive decoder fills the final 35 rows with black pixels.
Because those pixels cannot be recovered faithfully, this script removes that
single source row by its immutable byte hash instead of rewriting the image.

Without ``--push``, the cleaned DatasetDict is saved below
``--work-dir/mteb_export``. With ``--push``, it is uploaded to ``--repo-id``
and the resulting immutable Hub revision is printed.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Image, load_dataset, load_from_disk
from huggingface_hub import DatasetCard, HfApi, create_repo
from PIL import Image as PILImage
from PIL import ImageFile

SOURCE_REPO = "mteb/MMSoc_Memotion"
SOURCE_REVISION = "f77e225ae55c1987b0b8cbf6badd1c10296f5f34"
SOURCE_COUNTS = {"train": 5593, "validation": 699, "test": 700}
OUTPUT_COUNTS = {"train": 5592, "validation": 699, "test": 700}
CORRUPT_SPLIT = "train"
CORRUPT_INDEX = 4578
CORRUPT_FILENAME = "image_5119.png"
CORRUPT_SHA256 = "63175a3560ace1e74d4e7913206c94b6293473e09539df0e5be791df62e6a2a9"
EXPECTED_COLUMNS = [
    "image",
    "text_ocr",
    "text_corrected",
    "humor",
    "sarcasm",
    "offensive",
    "motivational",
    "sentiment",
    "split",
]


def _image_bytes(image: dict[str, Any]) -> tuple[bytes, str]:
    path = image.get("path")
    filename = Path(path).name if path else ""
    payload = image.get("bytes")
    if payload is not None:
        return bytes(payload), filename
    if path is None:
        raise ValueError("Image has neither embedded bytes nor a path")
    return Path(path).read_bytes(), filename


def _strict_decode(payload: bytes) -> None:
    with PILImage.open(BytesIO(payload)) as image:
        image.verify()
    with PILImage.open(BytesIO(payload)) as image:
        image.convert("RGB").load()


def _audit(dataset: DatasetDict) -> list[dict[str, str | int]]:
    corrupt: list[dict[str, str | int]] = []
    for split, split_dataset in dataset.items():
        for index, row in enumerate(split_dataset):
            payload, filename = _image_bytes(row["image"])
            try:
                _strict_decode(payload)
            except Exception as error:
                corrupt.append(
                    {
                        "split": split,
                        "index": index,
                        "filename": filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return corrupt


def _validate_source(dataset: DatasetDict) -> None:
    counts = {split: len(rows) for split, rows in dataset.items()}
    if counts != SOURCE_COUNTS:
        raise ValueError(f"Unexpected source split counts: {counts}")
    for split, rows in dataset.items():
        if rows.column_names != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {split}: {rows.column_names}")

    corrupt = _audit(dataset)
    expected = {
        "split": CORRUPT_SPLIT,
        "index": CORRUPT_INDEX,
        "filename": CORRUPT_FILENAME,
        "sha256": CORRUPT_SHA256,
    }
    if len(corrupt) != 1 or any(
        corrupt[0][key] != value for key, value in expected.items()
    ):
        raise ValueError(f"Unexpected corrupt images: {corrupt}")
    print(f"source={SOURCE_REPO}@{SOURCE_REVISION}")
    print(f"source_counts={counts}")
    print(f"corrupt={corrupt[0]}")


def _build_dataset(source: DatasetDict) -> DatasetDict:
    cleaned: dict[str, Dataset] = {}
    for split, rows in source.items():
        if split == CORRUPT_SPLIT:
            keep = [index for index in range(len(rows)) if index != CORRUPT_INDEX]
            rows = rows.select(keep)
        cleaned[split] = rows.cast_column("image", Image())
    return DatasetDict(cleaned)


def _validate_output(dataset: DatasetDict) -> None:
    counts = {split: len(rows) for split, rows in dataset.items()}
    if counts != OUTPUT_COUNTS:
        raise ValueError(f"Unexpected output split counts: {counts}")
    raw = DatasetDict(
        {
            split: rows.cast_column("image", Image(decode=False))
            for split, rows in dataset.items()
        }
    )
    corrupt = _audit(raw)
    if corrupt:
        raise ValueError(f"Output still contains corrupt images: {corrupt}")
    for split, rows in dataset.items():
        if rows.column_names != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {split}: {rows.column_names}")
    print(f"output_counts={counts}")
    print("strictly_decodable_images=6991 corrupt=0")


def _validate_preservation(source: DatasetDict, cleaned: DatasetDict) -> None:
    raw_cleaned = DatasetDict(
        {
            split: rows.cast_column("image", Image(decode=False))
            for split, rows in cleaned.items()
        }
    )
    for split, source_rows in source.items():
        source_indices = [
            index
            for index in range(len(source_rows))
            if not (split == CORRUPT_SPLIT and index == CORRUPT_INDEX)
        ]
        cleaned_rows = raw_cleaned[split]
        if len(source_indices) != len(cleaned_rows):
            raise ValueError(f"Unexpected preservation count in {split}")
        for cleaned_index, source_index in enumerate(source_indices):
            source_row = source_rows[source_index]
            cleaned_row = cleaned_rows[cleaned_index]
            source_payload, _ = _image_bytes(source_row["image"])
            cleaned_payload, _ = _image_bytes(cleaned_row["image"])
            if source_payload != cleaned_payload:
                raise ValueError(f"Image bytes changed in {split} row {source_index}")
            source_metadata = {
                key: value for key, value in source_row.items() if key != "image"
            }
            cleaned_metadata = {
                key: value for key, value in cleaned_row.items() if key != "image"
            }
            if source_metadata != cleaned_metadata:
                raise ValueError(f"Metadata changed in {split} row {source_index}")
    print("preserved_rows=6991 image_bytes_and_metadata_identical=true")


def _card_body() -> str:
    return f"""
# Memotion for MTEB

This repository is a byte-preserving cleanup of
[`{SOURCE_REPO}`](https://huggingface.co/datasets/{SOURCE_REPO}) at revision
`{SOURCE_REVISION}` for
[MTEB issue #5158](https://github.com/embeddings-benchmark/mteb/issues/5158).

The pinned source and the
[official Kaggle release](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k)
both contain the same truncated `train` image, `{CORRUPT_FILENAME}` (row
{CORRUPT_INDEX}, SHA-256 `{CORRUPT_SHA256}`). The PNG is missing part of its IDAT
payload. Permissive decoding fabricates 35 black rows, so this dataset excludes
that one example rather than altering its visual content.

All other source rows and embedded image bytes are preserved in their original
order. All 6,991 remaining images pass both Pillow verification and a complete
RGB decode. The validation and test splits are unchanged.

Generated by `scripts/data/memotion/create_data.py` in MTEB.
"""


def _save_local(dataset: DatasetDict, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    dataset.save_to_disk(output)
    reloaded = load_from_disk(output)
    if not isinstance(reloaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict after reload, got {type(reloaded)}")
    _validate_output(reloaded)
    print(f"saved={output}")


def _push(dataset: DatasetDict, repo_id: str) -> None:
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(
        repo_id,
        commit_message="Remove truncated Memotion image",
    )
    card = DatasetCard.load(repo_id, repo_type="dataset")
    generated_metadata = card.content.split("# Memotion for MTEB", maxsplit=1)[0]
    card.content = generated_metadata.rstrip() + "\n" + _card_body()
    card.push_to_hub(
        repo_id,
        repo_type="dataset",
        commit_message="Document Memotion image cleanup",
    )
    revision = HfApi().dataset_info(repo_id).sha
    print(f"pushed={repo_id}@{revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/memotion_mteb"),
        help="Directory for a local export when --push is omitted",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face dataset repository; required with --push",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.push and not args.repo_id:
        parser.error("--repo-id is required with --push")

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    source = load_dataset(SOURCE_REPO, revision=SOURCE_REVISION)
    source = DatasetDict(
        {
            split: rows.cast_column("image", Image(decode=False))
            for split, rows in source.items()
        }
    )
    _validate_source(source)
    cleaned = _build_dataset(source)
    _validate_output(cleaned)
    _validate_preservation(source, cleaned)

    if args.push:
        _push(cleaned, args.repo_id)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        _save_local(cleaned, args.work_dir / "mteb_export")


if __name__ == "__main__":
    main()
