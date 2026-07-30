#!/usr/bin/env python3
"""Package yale-nlp/MMVU multiple-choice examples for MTEB VideoCentricQA.

Keeps the public validation multiple-choice subset (625 examples, 379 videos),
resolves video URLs to local Hub files, and exports columns:
  question, video, candidates, answer.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/mmvu/create_data.py \\
      --repo-id {base_repo}/MMVU-VQA \\
      --work-dir /tmp/mmvu_mteb \\
      --push
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from datasets import Dataset, DatasetDict, Video, load_dataset
from huggingface_hub import create_repo, snapshot_download
from tqdm import tqdm

_SOURCE = "yale-nlp/MMVU"
_SOURCE_REVISION = "b937f414a87e9012acba49d95669020b24fa9ee9"
_VIDEO_URL_RE = re.compile(r"/videos/(.+\.mp4)$", re.IGNORECASE)
_CHOICE_KEYS = ("A", "B", "C", "D", "E")


def _video_relpath(url: str) -> str:
    path = unquote(urlparse(url).path)
    m = _VIDEO_URL_RE.search(path)
    if not m:
        raise ValueError(f"Unexpected MMVU video URL: {url}")
    return m.group(1)


def _build_rows(repo_dir: Path) -> list[dict]:
    ds = load_dataset(
        _SOURCE,
        revision=_SOURCE_REVISION,
        split="validation",
    )
    ds = ds.filter(lambda row: row["question_type"] == "multiple-choice")

    rows: list[dict] = []
    missing = 0
    for row in tqdm(ds, desc="rows"):
        rel = _video_relpath(row["video"])
        video_path = repo_dir / "videos" / rel
        if not video_path.is_file():
            missing += 1
            continue
        choices = row["choices"]
        candidates = [choices[k] for k in _CHOICE_KEYS]
        answer_letter = row["answer"]
        if answer_letter not in choices:
            raise SystemExit(f"Answer {answer_letter!r} not in choices for {row['id']}")
        rows.append(
            {
                "question_id": row["id"],
                "question": row["question"],
                "video": str(video_path),
                "candidates": candidates,
                "answer": choices[answer_letter],
                "subject": row["metadata"]["subject"],
            }
        )
    if missing:
        raise SystemExit(f"Missing {missing} video files under {repo_dir / 'videos'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="Wissam42/MMVU-VQA",
        help="Hub dataset id to push (or local export label)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/mmvu_mteb"),
        help="Cache / local export directory",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push DatasetDict test split to Hub",
    )
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {_SOURCE}@{_SOURCE_REVISION} …")
    repo_dir = Path(
        snapshot_download(
            repo_id=_SOURCE,
            repo_type="dataset",
            revision=_SOURCE_REVISION,
            local_dir=args.work_dir / "source",
        )
    )

    rows = _build_rows(repo_dir)
    print(f"Built {len(rows)} multiple-choice examples")
    ds = Dataset.from_list(rows).cast_column("video", Video())
    export = DatasetDict({"test": ds})

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        export.push_to_hub(args.repo_id, token=token)
        print(f"Pushed {args.repo_id}. Pin the Hub commit SHA in TaskMetadata.")
    else:
        out = args.work_dir / "mteb_export"
        export.save_to_disk(out)
        print(f"Wrote local dataset to {out} (pass --push to upload)")


if __name__ == "__main__":
    main()
