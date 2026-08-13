#!/usr/bin/env python3
"""Build a flat MTEB-format CaReBench video/caption dataset and push it to the Hub.

CaReBench (https://huggingface.co/datasets/MCG-NJU/CaReBench) ships as two loose
artifacts that `datasets` cannot join on its own:

  json/metadata.json   1,000 records, `video` holds a bare MP4 filename
  videos/videos.zip    1,000 MP4 files (~3.9 GB) + a duplicate metadata.json

The published config therefore exposes text only. This script downloads both
artifacts at a pinned revision, extracts the clips, asserts that the filenames
match 1:1 in both directions, and emits a single flat split where `video` is a
real `datasets.Video` feature, so the MTEB task can read it directly.

Row order follows `json/metadata.json`, which makes the derived row indices --
and therefore the qrels built from them in the task -- reproducible from the
pinned source revision.

Usage:
  # inspect the plan without touching the 3.9 GB archive
  uv run python scripts/data/carebench_retrieval/create_data.py --dry-run

  # build locally
  uv run python scripts/data/carebench_retrieval/create_data.py \\
      --work-dir /tmp/carebench_mteb

  # build and push
  export HF_TOKEN=...
  uv run python scripts/data/carebench_retrieval/create_data.py \\
      --work-dir /tmp/carebench_mteb \\
      --repo-id {your_namespace}/CaReBench \\
      --push
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Video
from huggingface_hub import create_repo, hf_hub_download
from tqdm import tqdm

SOURCE_REPO = "MCG-NJU/CaReBench"
# Pinned so the extracted clips and the row order stay reproducible.
SOURCE_REVISION = "8cf3e1d216791a1fe6a2a3383a0d077d62b7ff1c"

METADATA_FILE = "json/metadata.json"
VIDEOS_FILE = "videos/videos.zip"

EXPECTED_RECORDS = 1000

# Kept flat (no `id` column): the task assigns row-index ids, matching the
# existing VATEX / TUNA-Bench / MSVD video retrieval datasets in the mteb org.
TEXT_COLUMNS = [
    "caption",
    "spatial_caption",
    "temporal_caption",
    "category",
    "subcategory",
]


def _download_metadata() -> list[dict]:
    """Fetch json/metadata.json at the pinned revision (~3.7 MB)."""
    path = hf_hub_download(
        SOURCE_REPO,
        METADATA_FILE,
        repo_type="dataset",
        revision=SOURCE_REVISION,
    )
    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected a JSON array in {METADATA_FILE}, got {type(records)}"
        )
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records in {METADATA_FILE}, found {len(records)}"
        )

    missing = [c for c in ["video", *TEXT_COLUMNS] if c not in records[0]]
    if missing:
        raise ValueError(f"{METADATA_FILE} is missing expected columns: {missing}")

    return records


def _ensure_videos(work: Path) -> Path:
    """Download videos/videos.zip (~3.9 GB) and extract the MP4s into work/videos."""
    videos_root = work / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    marker = videos_root / ".extracted"
    if marker.exists():
        print(f"Videos already extracted in {videos_root}, skipping download")
        return videos_root

    local_zip = Path(
        hf_hub_download(
            SOURCE_REPO,
            VIDEOS_FILE,
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )

    with zipfile.ZipFile(local_zip) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".mp4")]
        for member in tqdm(members, desc="extract clips"):
            # Flatten to the bare filename: metadata.json refers to clips by
            # name only, and this also neutralises any path traversal.
            name = Path(member).name
            target = videos_root / name
            if target.exists():
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())

    marker.write_text("ok")
    return videos_root


def _verify_one_to_one(records: list[dict], videos_root: Path) -> None:
    """Assert the metadata filenames and the extracted clips match exactly."""
    wanted = [r["video"] for r in records]
    wanted_set = set(wanted)
    if len(wanted_set) != len(wanted):
        raise ValueError("metadata.json contains duplicate `video` filenames")

    found = {p.name for p in videos_root.glob("*.mp4")}

    missing = sorted(wanted_set - found)
    extra = sorted(found - wanted_set)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clip(s) referenced by metadata.json are missing, "
            f"e.g. {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"{len(extra)} extracted clip(s) are not referenced by metadata.json, "
            f"e.g. {extra[:5]}"
        )

    print(f"Verified 1:1 match on {len(wanted)} clips")


def _build_dataset(records: list[dict], videos_root: Path) -> Dataset:
    """Build the flat split, with `video` as a real Video feature."""
    data: dict[str, list] = {"video": []}
    for column in TEXT_COLUMNS:
        data[column] = []

    for record in records:
        data["video"].append(str(videos_root / record["video"]))
        for column in TEXT_COLUMNS:
            data[column].append(record[column])

    return Dataset.from_dict(data).cast_column("video", Video())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/carebench_mteb"))
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub dataset id to push to, e.g. {your_namespace}/CaReBench "
        "(required with --push)",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the pushed repo as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download metadata.json only and report the plan; never fetches "
        "the 3.9 GB video archive",
    )
    parser.add_argument(
        "--skip-video-download",
        action="store_true",
        help="Assume the clips are already extracted under work-dir/videos",
    )
    args = parser.parse_args()

    if args.push and not args.repo_id:
        raise SystemExit("--repo-id is required when using --push")

    records = _download_metadata()

    if args.dry_run:
        print(f"Source: {SOURCE_REPO}@{SOURCE_REVISION}")
        print(f"Records: {len(records)}")
        print(f"Columns: {['video', *TEXT_COLUMNS]}")
        print(f"Unique clips: {len({r['video'] for r in records})}")
        print(f"Would download {VIDEOS_FILE} (~3.9 GB) into {args.work_dir}")
        print("Dry run: no video archive fetched")
        return

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    videos_root = work / "videos" if args.skip_video_download else _ensure_videos(work)
    _verify_one_to_one(records, videos_root)

    ds = _build_dataset(records, videos_root)
    print(ds)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(
            args.repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=args.private,
        )
        DatasetDict({"test": ds}).push_to_hub(args.repo_id, token=token)
        print(
            f"Pushed to {args.repo_id}. Pin the commit SHA in "
            "TaskMetadata.dataset['revision']."
        )
    else:
        out = work / "mteb_export"
        out.mkdir(exist_ok=True)
        ds.save_to_disk(out)
        print(f"Wrote local dataset to {out}")


if __name__ == "__main__":
    main()
