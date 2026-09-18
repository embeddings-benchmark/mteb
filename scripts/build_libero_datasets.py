"""Construction script for LIBERO-I2V and LIBERO-V2I.

Builds image<->video retrieval tasks from physical-intelligence/libero
(LeRobot v2.0 snapshot of the LIBERO benchmark: 1,693 tabletop-manipulation
episodes over 40 tasks from the libero_spatial/object/goal/10 suites,
256x256 agentview frames at 10 fps).

Construction:
- Every episode is rendered to an H.264 mp4 (agentview stream only) and its
  final frame is kept as the goal-state image.
- Per task, episodes are deterministically split into a query pool
  (QUERIES_PER_TASK episodes) and a corpus pool (the rest), so a query's own
  source episode is never in the corpus and exact frame matching cannot
  solve the task.
- Relevance is task-level and multi-positive: a query matches every corpus
  episode of the same task. Distractors include same-scene episodes of
  other goals, so matching requires goal-state understanding rather than
  scene recognition.
- I2V: query = goal image of a query-pool episode, corpus = videos of
  corpus-pool episodes. V2I: query = video of a query-pool episode,
  corpus = goal images of corpus-pool episodes.

Usage:
    python build_libero_datasets.py --snapshot ~/MOEB/data/libero_pi \
        --workdir ~/MOEB/build/libero --hf-user <username> [--push]
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image as PILImage

from datasets import Dataset, DatasetDict, Image, Video

FPS = 10
QUERIES_PER_TASK = 10
CRF = 23


def episode_parquet(snapshot: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return snapshot / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def load_episode_frames(path: Path) -> list[bytes]:
    table = pq.read_table(path, columns=["image"])
    return [row["bytes"] for row in table.column("image").to_pylist()]


def encode_video(frames: list[bytes], out_path: Path) -> None:
    first = PILImage.open(io.BytesIO(frames[0]))
    width, height = first.size
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(CRF),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        img = PILImage.open(io.BytesIO(frame)).convert("RGB")
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--hf-user", type=str, required=True)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    video_dir = args.workdir / "videos"
    image_dir = args.workdir / "goal_images"
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(args.snapshot / "meta/episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    # Render every episode once; both directions reuse the artifacts.
    by_task: dict[str, list[int]] = defaultdict(list)
    for ep in episodes:
        idx = ep["episode_index"]
        task = ep["tasks"][0]
        by_task[task].append(idx)
        mp4 = video_dir / f"ep{idx:06d}.mp4"
        png = image_dir / f"ep{idx:06d}.png"
        if mp4.exists() and png.exists():
            continue
        frames = load_episode_frames(episode_parquet(args.snapshot, idx))
        encode_video(frames, mp4)
        goal = PILImage.open(io.BytesIO(frames[-1])).convert("RGB")
        goal.save(png)
        print(f"episode {idx}: {len(frames)} frames, task='{task}'")

    # Deterministic per-task split: lowest episode indices become queries.
    query_pool: dict[str, list[int]] = {}
    corpus_pool: dict[str, list[int]] = {}
    for task, idxs in sorted(by_task.items()):
        idxs = sorted(idxs)
        query_pool[task] = idxs[:QUERIES_PER_TASK]
        corpus_pool[task] = idxs[QUERIES_PER_TASK:]

    def build_direction(query_modality: str) -> dict[str, Dataset]:
        query_rows: list[dict] = []
        corpus_rows: list[dict] = []
        qrel_rows: list[dict] = []
        for task in sorted(by_task):
            for q in query_pool[task]:
                query_rows.append(
                    {
                        "id": f"q-ep{q:06d}",
                        "image": str(image_dir / f"ep{q:06d}.png"),
                        "video": str(video_dir / f"ep{q:06d}.mp4"),
                    }
                )
                for c in corpus_pool[task]:
                    qrel_rows.append(
                        {
                            "query-id": f"q-ep{q:06d}",
                            "corpus-id": f"c-ep{c:06d}",
                            "score": 1,
                        }
                    )
            for c in corpus_pool[task]:
                corpus_rows.append(
                    {
                        "id": f"c-ep{c:06d}",
                        "image": str(image_dir / f"ep{c:06d}.png"),
                        "video": str(video_dir / f"ep{c:06d}.mp4"),
                    }
                )

        doc_modality = "video" if query_modality == "image" else "image"
        queries = Dataset.from_list(
            [{k: r[k] for k in ("id", query_modality)} for r in query_rows]
        ).cast_column(query_modality, Video() if query_modality == "video" else Image())
        corpus = Dataset.from_list(
            [{k: r[k] for k in ("id", doc_modality)} for r in corpus_rows]
        ).cast_column(doc_modality, Video() if doc_modality == "video" else Image())
        qrels = Dataset.from_list(qrel_rows)
        return {"queries": queries, "corpus": corpus, "qrels": qrels}

    for direction, query_modality in (("I2V", "image"), ("V2I", "video")):
        parts = build_direction(query_modality)
        repo = f"{args.hf_user}/LIBERO-{direction}"
        for config_name, ds in parts.items():
            print(f"{repo} [{config_name}]: {len(ds)} rows")
            if args.push:
                DatasetDict({"test": ds}).push_to_hub(repo, config_name=config_name)


if __name__ == "__main__":
    main()
