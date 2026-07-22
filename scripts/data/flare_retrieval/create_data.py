#!/usr/bin/env python3
"""Build subsampled MTEB-format FLARE text→clip retrieval datasets and push.

Reads annotations from base FLARE dataset and clips from the accompanying video
zips. Builds a seeded eval subset (default 1k queries / up to 4k corpus clips).

Usage:
  export HF_TOKEN=...

  python scripts/data/flare_retrieval/create_data.py \\
      --direction unified \\
      --repo-id {base_repo_id}/FLARE-1k-Unified-T2VA \\
      --work-dir /tmp/flare_mteb \\
      --push

  python scripts/data/flare_retrieval/create_data.py \\
      --direction vision \\
      --repo-id {base_repo_id}/FLARE-1k-Vision-T2V \\
      --work-dir /tmp/flare_mteb \\
      --push

  python scripts/data/flare_retrieval/create_data.py \\
      --direction audio \\
      --repo-id {base_repo_id}/FLARE-1k-Audio-T2VA \\
      --work-dir /tmp/flare_mteb \\
      --push
"""

from __future__ import annotations

import argparse
import os
import random
import zipfile
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, Video, load_dataset
from huggingface_hub import create_repo, hf_hub_download, list_repo_files
from tqdm import tqdm

DIRECTION_CONFIG = {
    "unified": {
        "query_config": "clip-query-unified",
        "text_key": "unified_caption",
        "include_audio": True,
    },
    "vision": {
        "query_config": "clip-query-vision",
        "text_key": "caption",
        "include_audio": False,
    },
    "audio": {
        "query_config": "clip-query-audio",
        "text_key": "audio_caption",
        "include_audio": True,
    },
}


def _ensure_videos(work: Path) -> Path:
    """Download and extract FLARE video zips into work/videos."""
    videos_root = work / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    files = list_repo_files("YqjMartin/FLARE", repo_type="dataset")
    zips = sorted(f for f in files if f.startswith("videos/") and f.endswith(".zip"))
    for rel in tqdm(zips, desc="download/extract video zips"):
        local_zip = Path(hf_hub_download("YqjMartin/FLARE", rel, repo_type="dataset"))
        # Skip extract if a marker exists for this zip
        marker = videos_root / f".extracted_{local_zip.stem}"
        if marker.exists():
            continue
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(videos_root)
        marker.write_text("ok")
    return videos_root


def _resolve_clip(videos_root: Path, video_path: str) -> Path:
    # Annotations use e.g. "<vid>/<vid>-Scene-067.mp4" (or .wav for audio captions)
    mp4 = video_path.replace(".wav", ".mp4")
    candidates = [
        videos_root / mp4,
        videos_root / Path(mp4).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to recursive search by filename
    matches = list(videos_root.rglob(Path(mp4).name))
    if not matches:
        raise FileNotFoundError(f"Clip not found for {video_path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction", choices=sorted(DIRECTION_CONFIG), default="unified"
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub dataset id (required with --push)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/flare_mteb"))
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--max-corpus", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--skip-video-download",
        action="store_true",
        help="Assume videos already extracted under work-dir/videos",
    )
    args = parser.parse_args()

    if args.push and not args.repo_id:
        raise SystemExit("--repo-id is required when using --push")

    cfg = DIRECTION_CONFIG[args.direction]
    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    videos_root = work / "videos" if args.skip_video_download else _ensure_videos(work)

    ds = load_dataset("YqjMartin/FLARE", cfg["query_config"], split="test")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.num_queries]

    queries = {"id": [], "text": []}
    qrels = {"query-id": [], "corpus-id": [], "score": []}
    needed_clips: set[str] = set()

    for i, idx in enumerate(tqdm(indices, desc="queries")):
        row = ds[idx]
        text = row[cfg["text_key"]]
        video_path = row["video_path"]
        # Normalize to mp4 path id
        corpus_id = video_path.replace(".wav", ".mp4")
        qid = f"q-{i}"
        queries["id"].append(qid)
        queries["text"].append(text)
        qrels["query-id"].append(qid)
        qrels["corpus-id"].append(corpus_id)
        qrels["score"].append(1)
        needed_clips.add(corpus_id)

    # Add seeded distractors from the same annotation file
    all_clips = sorted({row["video_path"].replace(".wav", ".mp4") for row in ds})
    distractors = [c for c in all_clips if c not in needed_clips]
    rng.shuffle(distractors)
    for c in distractors:
        if len(needed_clips) >= args.max_corpus:
            break
        needed_clips.add(c)

    corpus = {"id": [], "video": []}
    if cfg["include_audio"]:
        corpus["audio"] = []

    for cid in tqdm(sorted(needed_clips), desc="corpus"):
        clip_path = _resolve_clip(videos_root, cid)
        corpus["id"].append(cid)
        corpus["video"].append(str(clip_path))
        if cfg["include_audio"]:
            corpus["audio"].append(str(clip_path))

    corpus_ds = Dataset.from_dict(corpus).cast_column("video", Video())
    if cfg["include_audio"]:
        corpus_ds = corpus_ds.cast_column("audio", Audio())
    queries_ds = Dataset.from_dict(queries)
    qrels_ds = Dataset.from_dict(qrels)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus_ds}).push_to_hub(
            args.repo_id, "corpus", token=token
        )
        DatasetDict({"test": queries_ds}).push_to_hub(
            args.repo_id, "queries", token=token
        )
        DatasetDict({"test": qrels_ds}).push_to_hub(args.repo_id, "qrels", token=token)
        print(
            f"Pushed to {args.repo_id}. Pin the commit SHA in "
            "TaskMetadata.dataset['revision']."
        )
    else:
        out = work / f"mteb_export_{args.direction}"
        out.mkdir(exist_ok=True)
        corpus_ds.save_to_disk(out / "corpus")
        queries_ds.save_to_disk(out / "queries")
        qrels_ds.save_to_disk(out / "qrels")
        print(f"Wrote local datasets to {out}")


if __name__ == "__main__":
    main()
