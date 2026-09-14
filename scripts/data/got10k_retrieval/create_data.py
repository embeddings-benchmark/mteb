#!/usr/bin/env python3
"""Build the bidirectional GOT-10k image/video retrieval tasks for MTEB.

GOT-10k (Generic Object Tracking benchmark) contains 180 validation sequences
with per-frame ground-truth bounding boxes. Each sequence is one tracked object
across a real-world video clip. MTEB uses the validation split because the test
split has no public annotations.

For I2V: query = first frame (image) of a sequence → retrieve the tracking video.
For V2I: query = tracking video → retrieve the first frame (image).

The mapping is one-to-one: each query has exactly one relevant item.

Download source:
  Register and download val.zip from http://got-10k.aitestunion.com/downloads
  Then extract it:
    unzip val.zip -d /path/to/got10k_val

Examples:
  # Inspect the extracted val directory (no push).
  uv run python scripts/data/got10k_retrieval/create_data.py \\
      --val-dir /path/to/got10k_val --dry-run

  # Build locally and push to HuggingFace.
  uv run python scripts/data/got10k_retrieval/create_data.py \\
      --val-dir /path/to/got10k_val --push

  # Push only the i2v direction.
  uv run python scripts/data/got10k_retrieval/create_data.py \\
      --val-dir /path/to/got10k_val --push --direction i2v
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, DatasetDict, Image, Value, Video
from huggingface_hub import HfApi, create_repo, get_token
from PIL import Image as PILImage

_LICENSE = "cc-by-4.0"
_REFERENCE = "https://arxiv.org/abs/1808.00803"
_I2V_REPO_DEFAULT = "rakshi719/GOT10k-I2V"
_V2I_REPO_DEFAULT = "rakshi719/GOT10k-V2I"

Direction = Literal["v2i", "i2v"]


def _find_sequences(val_dir: Path) -> list[Path]:
    """Return sorted list of sequence directories in val_dir."""
    seqs = sorted(
        p for p in val_dir.iterdir() if p.is_dir() and p.name.startswith("GOT-10k_Val_")
    )
    if not seqs:
        raise RuntimeError(f"No GOT-10k_Val_* directories found in {val_dir}")
    return seqs


def _read_meta(seq_dir: Path) -> dict[str, str]:
    """Read meta_info.ini and return key→value dict."""
    meta_path = seq_dir / "meta_info.ini"
    if not meta_path.is_file():
        return {}
    cfg = configparser.ConfigParser()
    cfg.read(meta_path, encoding="utf-8")
    result = {}
    for section in cfg.sections():
        for key, val in cfg.items(section):
            result[key] = val
    return result


def _frames(seq_dir: Path) -> list[Path]:
    """Return sorted list of frame JPEGs in a sequence directory."""
    return sorted(seq_dir.glob("*.jpg"))


def _get_ffmpeg() -> str:
    """Return path to ffmpeg, preferring system install then imageio-ffmpeg bundle."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found; install with: brew install ffmpeg  OR  pip install imageio-ffmpeg"
    )


def _frames_to_mp4(frames: list[Path], output_path: Path, fps: int = 10) -> None:
    """Convert a list of JPEG frames to an MP4 video using ffmpeg."""
    ffmpeg = _get_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            (tmp_dir / f"{i:08d}.jpg").symlink_to(frame.resolve())

        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate", str(fps),
                "-i", str(tmp_dir / "%08d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {output_path}:\n{result.stderr}"
            )


def _build_sequences(
    val_dir: Path,
    video_dir: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Process all sequences: extract first frame + encode MP4. Return metadata list."""
    seqs = _find_sequences(val_dir)
    print(f"Found {len(seqs)} sequences")

    records = []
    for seq_dir in seqs:
        seq_id = seq_dir.name
        frames = _frames(seq_dir)
        if not frames:
            print(f"  SKIP {seq_id}: no frames found")
            continue

        meta = _read_meta(seq_dir)
        first_frame = frames[0]
        video_path = video_dir / f"{seq_id}.mp4"

        if not dry_run and not video_path.is_file():
            print(f"  Encoding {seq_id} ({len(frames)} frames) → {video_path.name}")
            _frames_to_mp4(frames, video_path)
        elif dry_run:
            print(f"  [dry-run] {seq_id}: {len(frames)} frames, meta={list(meta.keys())[:4]}")

        records.append({
            "seq_id": seq_id,
            "first_frame": str(first_frame),
            "video_path": str(video_path),
            "n_frames": len(frames),
            "object_class": meta.get("object_class", ""),
            "motion_type": meta.get("motion_type", ""),
        })

    return records


def _build_i2v_datasets(records: list[dict[str, Any]]) -> tuple[Dataset, Dataset, Dataset]:
    """Build I2V: queries=images (first frames), corpus=videos."""
    queries = Dataset.from_dict({
        "id": [r["seq_id"] for r in records],
        "image": [r["first_frame"] for r in records],
    }).cast_column("image", Image())

    corpus = Dataset.from_dict({
        "id": [r["seq_id"] for r in records],
        "video": [r["video_path"] for r in records],
    }).cast_column("video", Video())

    # one-to-one qrels: each image query → its own video
    qrels = Dataset.from_dict({
        "query-id": [r["seq_id"] for r in records],
        "corpus-id": [r["seq_id"] for r in records],
        "score": [1] * len(records),
    }).cast_column("score", Value("int32"))

    return corpus, queries, qrels


def _build_v2i_datasets(records: list[dict[str, Any]]) -> tuple[Dataset, Dataset, Dataset]:
    """Build V2I: queries=videos, corpus=images (first frames)."""
    queries = Dataset.from_dict({
        "id": [r["seq_id"] for r in records],
        "video": [r["video_path"] for r in records],
    }).cast_column("video", Video())

    corpus = Dataset.from_dict({
        "id": [r["seq_id"] for r in records],
        "image": [r["first_frame"] for r in records],
    }).cast_column("image", Image())

    qrels = Dataset.from_dict({
        "query-id": [r["seq_id"] for r in records],
        "corpus-id": [r["seq_id"] for r in records],
        "score": [1] * len(records),
    }).cast_column("score", Value("int32"))

    return corpus, queries, qrels


def _dataset_card(direction: Direction, n_seqs: int) -> str:
    if direction == "i2v":
        pretty_name = "GOT-10k Image-to-Video Retrieval"
        direction_tag = "image-to-video"
        query_desc = "first-frame images"
        corpus_desc = "tracking videos"
        task_desc = "given the first frame of a tracking sequence, retrieve the full tracking video"
    else:
        pretty_name = "GOT-10k Video-to-Image Retrieval"
        direction_tag = "video-to-image"
        query_desc = "tracking videos"
        corpus_desc = "first-frame images"
        task_desc = "given a tracking video, retrieve its corresponding first frame"

    return f"""---
license: {_LICENSE}
pretty_name: {pretty_name}
tags:
- mteb
- moeb
- {direction_tag}
- cross-modal-retrieval
- object-tracking
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
---

# {pretty_name}

MTEB/MOEB representation of the GOT-10k validation split for {direction_tag} retrieval.

## Task

{task_desc.capitalize()}. The mapping is one-to-one: each query has exactly one
relevant item (the other direction of the same sequence).

## Contents

- **Queries**: {n_seqs} {query_desc}
- **Corpus**: {n_seqs} {corpus_desc}
- **Qrels**: {n_seqs} one-to-one binary relevance judgments

## Source

GOT-10k (Generic Object Tracking benchmark) validation split — 180 real-world
sequences spanning 563 object classes and 87 motion patterns. Videos are encoded
from the official JPEG frames at 10 fps with libx264.

Paper: {_REFERENCE}
Official site: http://got-10k.aitestunion.com/

## License

CC-BY-4.0. See the [official GOT-10k page](http://got-10k.aitestunion.com/) for details.

## Citation

```bibtex
@article{{huang2019got,
  author = {{Lianghua Huang and Xin Zhao and Kaiqi Huang}},
  title = {{GOT-10k: A Large High-Diversity Benchmark for Generic Object
           Tracking in the Wild}},
  journal = {{IEEE Transactions on Pattern Analysis and Machine Intelligence}},
  year = {{2019}},
}}
```
"""


def _publish(
    repo_id: str,
    direction: Direction,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    work_dir: Path,
    n_seqs: int,
) -> str:
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("No HuggingFace token found; run `huggingface-cli login`")

    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=_dataset_card(direction, n_seqs).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {direction} dataset card",
    )
    DatasetDict({"test": corpus}).push_to_hub(
        repo_id, "corpus", token=token, max_shard_size="500MB",
        commit_message=f"Add {direction} corpus",
    )
    DatasetDict({"test": queries}).push_to_hub(
        repo_id, "queries", token=token, max_shard_size="500MB",
        commit_message=f"Add {direction} queries",
    )
    DatasetDict({"test": qrels}).push_to_hub(
        repo_id, "qrels", token=token,
        commit_message="Add relevance judgments",
    )
    revision = api.dataset_info(repo_id).sha
    (work_dir / f"hub_revision_{direction}.txt").write_text(
        f"{revision}\n", encoding="utf-8"
    )
    print(f"Pushed {repo_id} @ {revision}")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-dir", type=Path, required=True,
        help="Path to extracted GOT-10k val directory (contains GOT-10k_Val_* subdirs)",
    )
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path("/tmp/got10k_retrieval"),
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--i2v-repo-id", default=_I2V_REPO_DEFAULT,
        help="HuggingFace repo for I2V direction",
    )
    parser.add_argument(
        "--v2i-repo-id", default=_V2I_REPO_DEFAULT,
        help="HuggingFace repo for V2I direction",
    )
    parser.add_argument(
        "--direction", choices=("i2v", "v2i", "both"), default="both",
    )
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Skip encoding, just inspect")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    video_dir = work_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    val_dir = args.val_dir.resolve()
    if not val_dir.is_dir():
        raise RuntimeError(f"val-dir not found: {val_dir}")

    records = _build_sequences(val_dir, video_dir, dry_run=args.dry_run)
    print(json.dumps({"sequences": len(records)}, indent=2))

    if args.dry_run:
        print("Dry run complete — no datasets built or pushed.")
        return

    directions: list[Direction] = (
        ["i2v", "v2i"] if args.direction == "both" else [args.direction]  # type: ignore[list-item]
    )
    repo_ids: dict[Direction, str] = {
        "i2v": args.i2v_repo_id,
        "v2i": args.v2i_repo_id,
    }

    for direction in directions:
        print(f"\nBuilding {direction} datasets...")
        if direction == "i2v":
            corpus, queries, qrels = _build_i2v_datasets(records)
        else:
            corpus, queries, qrels = _build_v2i_datasets(records)

        print(f"  queries: {len(queries)}, corpus: {len(corpus)}, qrels: {len(qrels)}")

        if args.push:
            _publish(repo_ids[direction], direction, corpus, queries, qrels, work_dir, len(records))
        else:
            export_dir = work_dir / "export" / direction
            export_dir.mkdir(parents=True, exist_ok=True)
            DatasetDict({"test": corpus}).save_to_disk(export_dir / "corpus")
            DatasetDict({"test": queries}).save_to_disk(export_dir / "queries")
            DatasetDict({"test": qrels}).save_to_disk(export_dir / "qrels")
            print(f"  Saved to {export_dir} (use --push to upload to HuggingFace)")


if __name__ == "__main__":
    main()
