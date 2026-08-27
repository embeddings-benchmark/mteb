"""Build the reduced GLAMI-1M artifact from a verified official archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value

SOURCE_REVISION = "befda45d8d4e8b8082bb8a1912d1f9eb9483991c"
ARCHIVE_ROOT = "GLAMI-1M-dataset"
ARCHIVE_BYTES = 11_227_139_414
ARCHIVE_MD5 = "500348bbf54595db81cba353acd50d78"
ARCHIVE_SHA256 = "35cb7560d150b2147d9bc7813471036f08ac290fe8e17b16c7b08747de9026f7"
EXPECTED_ROWS = {"train": 1_000_000, "test": 116_004}
EXPECTED_GEOS = {
    "bg",
    "cz",
    "ee",
    "es",
    "gr",
    "hr",
    "hu",
    "lt",
    "lv",
    "ro",
    "si",
    "sk",
    "tr",
}
EXPECTED_SELECTION_SHA256 = (
    "d2efa971672220f160bbb8eea69e0bead024a92686ba18620866cef0a4a04e03"
)
TRAIN_PER_LABEL = 80
SEED = 42
FEATURES = Features(
    {"image": Image(), "text": Value("string"), "label": Value("int64")}
)


def _archive_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _verify_archive(path: Path) -> None:
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Unexpected GLAMI-1M archive size")
    if _archive_hashes(path) != (ARCHIVE_MD5, ARCHIVE_SHA256):
        raise ValueError("Unexpected GLAMI-1M archive checksum")


def _validated_member_path(info: ZipInfo) -> Path:
    name = info.filename
    stripped_name = name[:-1] if name.endswith("/") else name
    parts = stripped_name.split("/")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)

    if (
        not stripped_name
        or "\\" in name
        or ":" in name
        or "\x00" in name
        or any(part in {"", ".", ".."} for part in parts)
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or parts[0] != ARCHIVE_ROOT
    ):
        raise BadZipFile(f"Unsafe archive member path: {name!r}")
    if info.flag_bits & 0x1:
        raise BadZipFile(f"Encrypted archive member is not supported: {name!r}")
    if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise BadZipFile(f"Unsupported archive member type: {name!r}")
    return Path(*parts)


def _validate_members(archive: ZipFile) -> None:
    seen = set()
    for info in archive.infolist():
        path = _validated_member_path(info)
        normalized = path.as_posix()
        if normalized in seen:
            raise BadZipFile(f"Duplicate archive member: {info.filename!r}")
        seen.add(normalized)


def _extract_members(archive: ZipFile, names: set[str], destination: Path) -> None:
    for name in sorted(names):
        try:
            info = archive.getinfo(name)
        except KeyError as error:
            raise BadZipFile(f"Missing archive member: {name!r}") from error

        target = destination / _validated_member_path(info)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        crc = 0
        with archive.open(info) as source, target.open("xb") as output:
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
                crc = zlib.crc32(chunk, crc)
        if crc != info.CRC:
            raise BadZipFile(f"CRC mismatch for archive member: {name!r}")


def _scan(path: Path) -> tuple[list[int], set[str], set[str]]:
    labels = []
    geos = set()
    image_ids = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["category"]))
            geos.add(row["geo"])
            image_ids.add(row["image_id"])
    return labels, geos, image_ids


def _select_train(labels: list[int]) -> list[int]:
    indices = np.arange(len(labels))
    np.random.RandomState(SEED).shuffle(indices)
    counts: Counter[int] = Counter()
    selected = []
    for index in indices:
        label = labels[int(index)]
        if counts[label] < TRAIN_PER_LABEL:
            selected.append(int(index))
            counts[label] += 1
    return sorted(selected)


def _selected_image_ids(path: Path, selected: set[int]) -> set[str]:
    image_ids = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index in selected:
                image_ids.add(row["image_id"])
    return image_ids


def _examples(
    csv_path: Path, images_dir: Path, selected: set[int] | None
) -> Iterator[dict[str, str | int]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if selected is not None and index not in selected:
                continue
            image_path = images_dir / f"{row['image_id']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            yield {
                "image": str(image_path),
                "text": f"{row['name']}\n{row['description'] or ''}".strip(),
                "label": int(row["category"]),
            }


def _build_from_extraction(root: Path) -> tuple[DatasetDict, dict]:
    scans = {
        split: _scan(root / f"GLAMI-1M-{split}.csv") for split in ("train", "test")
    }
    for split, (labels, geos, _) in scans.items():
        if len(labels) != EXPECTED_ROWS[split]:
            raise ValueError(f"Unexpected {split} row count")
        if len(set(labels)) != 191 or geos != EXPECTED_GEOS:
            raise ValueError(f"Unexpected {split} labels or geographies")
    if scans["train"][2] & scans["test"][2]:
        raise ValueError("Train and test image IDs overlap")

    train_indices = _select_train(scans["train"][0])
    selection_hash = hashlib.sha256(
        json.dumps(train_indices, separators=(",", ":")).encode()
    ).hexdigest()
    if len(train_indices) != 15_228 or selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError("Training selection is not reproducible")

    images_dir = root / "images"
    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _examples,
                features=FEATURES,
                gen_kwargs={
                    "csv_path": root / "GLAMI-1M-train.csv",
                    "images_dir": images_dir,
                    "selected": set(train_indices),
                },
            ),
            "test": Dataset.from_generator(
                _examples,
                features=FEATURES,
                gen_kwargs={
                    "csv_path": root / "GLAMI-1M-test.csv",
                    "images_dir": images_dir,
                    "selected": None,
                },
            ),
        }
    )
    manifest = {
        "source_revision": SOURCE_REVISION,
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": ARCHIVE_SHA256,
        "selection_seed": SEED,
        "max_train_rows_per_label": TRAIN_PER_LABEL,
        "selection_sha256": selection_hash,
        "rows": {split: len(data) for split, data in dataset.items()},
    }
    return dataset, manifest


def build(archive_path: Path, work_dir: Path) -> tuple[DatasetDict, dict]:
    """Verify the archive, safely extract required rows, and build the artifact."""
    _verify_archive(archive_path)
    with ZipFile(archive_path) as archive:
        _validate_members(archive)
        csv_members = {
            f"{ARCHIVE_ROOT}/GLAMI-1M-train.csv",
            f"{ARCHIVE_ROOT}/GLAMI-1M-test.csv",
        }
        _extract_members(archive, csv_members, work_dir)

        root = work_dir / ARCHIVE_ROOT
        train_labels, _, train_image_ids = _scan(root / "GLAMI-1M-train.csv")
        _, _, test_image_ids = _scan(root / "GLAMI-1M-test.csv")
        train_indices = set(_select_train(train_labels))
        selected_train_image_ids = _selected_image_ids(
            root / "GLAMI-1M-train.csv", train_indices
        )
        image_ids = selected_train_image_ids | test_image_ids
        if not selected_train_image_ids <= train_image_ids:
            raise ValueError("Selected training image IDs are invalid")
        image_members = {
            f"{ARCHIVE_ROOT}/images/{image_id}.jpg" for image_id in image_ids
        }
        _extract_members(archive, image_members, work_dir)

    return _build_from_extraction(root)


@contextmanager
def _new_work_dir(path: Path | None) -> Iterator[Path]:
    if path is not None:
        path.mkdir(parents=True, exist_ok=False)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="mteb-glami-1m-") as directory:
        yield Path(directory)


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="new directory for the verified extraction (default: temporary)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--visibility",
        choices=("private", "public"),
        default="private",
        help="Hub repository visibility when pushing (default: private)",
    )
    parser.add_argument("--num-proc", type=int)
    args = parser.parse_args()
    if args.output_dir is None and not args.push:
        parser.error("provide --output-dir or --push")
    if args.push and not args.repo_id:
        parser.error("--repo-id is required with --push")
    if (
        args.work_dir is not None
        and args.output_dir is not None
        and _paths_overlap(args.work_dir, args.output_dir)
    ):
        parser.error("--work-dir and --output-dir must not overlap")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with _new_work_dir(args.work_dir) as work_dir:
        dataset, manifest = build(args.archive, work_dir)
        if args.output_dir is not None:
            dataset.save_to_disk(args.output_dir, num_proc=args.num_proc)
            (args.output_dir / "mteb-manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        if args.push:
            dataset.push_to_hub(
                args.repo_id,
                private=args.visibility == "private",
                max_shard_size="500MB",
                num_proc=args.num_proc,
            )
    print(json.dumps(manifest, indent=2))  # noqa: T201
