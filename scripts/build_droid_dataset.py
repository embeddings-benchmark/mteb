"""Construction script for DROID-IT2V.

Builds an image+text -> video retrieval task from cadene/droid_1.0.1
(LeRobot v2.1 conversion of DROID: 95,600 real-world Franka manipulation
episodes with language instructions and three synchronized camera streams).

Construction:
- Episodes are filtered to those with a non-empty language instruction and
  5-60 s duration, deduplicated by instruction (one episode per unique
  instruction), and evenly subsampled over the episode index range. A
  candidate is kept only if its `is_episode_successful` flag is true, until
  TARGET_EPISODES pass.
- Query = the FIRST frame of the exterior_1 camera (initial scene) plus the
  language instruction. Corpus document = the full episode video from the
  exterior_2 camera. Query and document come from different viewpoints of
  the same episode, so exact frame matching cannot solve the task; the
  model must ground "what will happen here" across viewpoints.
- Relevance is instance-level 1:1 (each query matches exactly its own
  episode's exterior_2 video).

Usage:
    python build_droid_dataset.py --workdir ~/MOEB/build/droid \
        --hf-user <username> [--phase sample|build] [--push]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from datasets import Dataset, DatasetDict, Image, Video

SOURCE_REPO = "cadene/droid_1.0.1"
TARGET_EPISODES = 1500
CANDIDATE_POOL = 2400
MIN_FRAMES, MAX_FRAMES = 75, 900  # 5-60 s at 15 fps
QUERY_CAM = "observation.images.exterior_1_left"
CORPUS_CAM = "observation.images.exterior_2_left"


def hub_path(kind: str, episode_index: int, camera: str | None = None) -> str:
    chunk = episode_index // 1000
    if kind == "parquet":
        return f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    return f"videos/chunk-{chunk:03d}/{camera}/episode_{episode_index:06d}.mp4"


def fetch(filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            SOURCE_REPO, filename, repo_type="dataset", cache_dir=str(cache_dir)
        )
    )


def episode_is_successful(parquet_path: Path) -> bool:
    table = pq.read_table(parquet_path, columns=["is_episode_successful"])
    values = table.column("is_episode_successful").to_pylist()
    return bool(values and values[-1])


def select_candidates(meta_dir: Path) -> list[dict]:
    with open(meta_dir / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    seen_instructions: set[str] = set()
    unique: list[dict] = []
    for ep in episodes:
        instruction = ep["tasks"][0].strip()
        if not instruction or instruction.lower() in seen_instructions:
            continue
        if not (MIN_FRAMES <= ep["length"] <= MAX_FRAMES):
            continue
        seen_instructions.add(instruction.lower())
        unique.append(
            {"episode_index": ep["episode_index"], "instruction": instruction}
        )
    # Deterministic even spread across the collection (sites/labs are
    # ordered by episode index, so this also spreads scenes and buildings).
    step = max(1, len(unique) // CANDIDATE_POOL)
    return unique[::step][:CANDIDATE_POOL]


def phase_sample(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    meta_dir = args.workdir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl",):
        target = meta_dir / name
        if not target.exists():
            target.write_bytes(fetch(f"meta/{name}", cache_dir).read_bytes())

    candidates = select_candidates(meta_dir)
    print(f"{len(candidates)} candidates after filtering/dedup")

    def check(cand: dict) -> dict | None:
        try:
            parquet = fetch(hub_path("parquet", cand["episode_index"]), cache_dir)
            return cand if episode_is_successful(parquet) else None
        except Exception as exc:  # noqa: BLE001 - skip broken episodes
            print(f"skip ep{cand['episode_index']}: {exc}")
            return None

    successful: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(check, c) for c in candidates]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                successful.append(result)
    successful = sorted(successful, key=lambda c: c["episode_index"])
    print(f"{len(successful)} successful episodes; fetching videos")

    def grab_videos(cand: dict) -> dict | None:
        # Some source episodes are missing one of the camera mp4s; drop them.
        try:
            for cam in (QUERY_CAM, CORPUS_CAM):
                fetch(hub_path("video", cand["episode_index"], cam), cache_dir)
            return cand
        except Exception as exc:  # noqa: BLE001
            print(f"drop ep{cand['episode_index']} (missing video): {exc}")
            return None

    accepted: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(grab_videos, successful):
            if result is not None:
                accepted.append(result)
            if len(accepted) >= TARGET_EPISODES:
                break
    accepted = sorted(accepted, key=lambda c: c["episode_index"])[:TARGET_EPISODES]
    print(f"{len(accepted)} episodes with both videos accepted")

    with open(args.workdir / "accepted.jsonl", "w") as f:
        for cand in accepted:
            f.write(json.dumps(cand) + "\n")
    print("sample phase done")


def phase_build(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    image_dir = args.workdir / "query_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with open(args.workdir / "accepted.jsonl") as f:
        accepted = [json.loads(line) for line in f]

    query_rows, corpus_rows, qrel_rows = [], [], []
    for cand in accepted:
        idx = cand["episode_index"]
        query_mp4 = fetch(hub_path("video", idx, QUERY_CAM), cache_dir)
        corpus_mp4 = fetch(hub_path("video", idx, CORPUS_CAM), cache_dir)
        png = image_dir / f"ep{idx:06d}.png"
        if not png.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(query_mp4),
                    "-frames:v",
                    "1",
                    str(png),
                ],
                check=True,
            )
        query_rows.append(
            {"id": f"q-ep{idx:06d}", "text": cand["instruction"], "image": str(png)}
        )
        corpus_rows.append({"id": f"c-ep{idx:06d}", "video": str(corpus_mp4)})
        qrel_rows.append(
            {"query-id": f"q-ep{idx:06d}", "corpus-id": f"c-ep{idx:06d}", "score": 1}
        )

    queries = Dataset.from_list(query_rows).cast_column("image", Image())
    corpus = Dataset.from_list(corpus_rows).cast_column("video", Video())
    qrels = Dataset.from_list(qrel_rows)
    repo = f"{args.hf_user}/DROID-IT2V"
    for config_name, ds in (("queries", queries), ("corpus", corpus), ("qrels", qrels)):
        print(f"{repo} [{config_name}]: {len(ds)} rows")
        if args.push:
            DatasetDict({"test": ds}).push_to_hub(repo, config_name=config_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--hf-user", type=str, required=True)
    parser.add_argument("--phase", choices=["sample", "build"], default="sample")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.phase == "sample":
        phase_sample(args)
    else:
        phase_build(args)


if __name__ == "__main__":
    main()
