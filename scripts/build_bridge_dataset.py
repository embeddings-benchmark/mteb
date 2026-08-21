"""Construction script for BridgeData-V2V.

Builds a cross-viewpoint video->video retrieval task from
IPEC-COMMUNITY/bridge_orig_lerobot (LeRobot v2.0 conversion of BridgeData
V2: 53,192 real-world WidowX manipulation episodes, up to four camera
streams at 5 fps).

Construction:
- Only a subset of episodes carries a real second viewpoint (the other
  camera slots hold ~2 KB placeholder clips). Episodes are selected by
  listing the per-file sizes on the Hub and keeping those whose image_0
  AND image_1 mp4s are both larger than SIZE_THRESHOLD.
- Episodes are filtered to 3-60 s, deduplicated by language instruction
  (one episode per unique instruction, which also spreads scenes and
  labs), and evenly subsampled over the episode index range.
- Query = the full image_1 video (alternate viewpoint). Corpus document =
  the full image_0 video (main over-shoulder viewpoint) of the same
  episode. Relevance is instance-level 1:1; queries and documents never
  share a viewpoint, so exact frame matching cannot solve the task.

Usage:
    python build_bridge_dataset.py --workdir ~/MOEB/build/bridge \
        --hf-user <username> [--phase sample|build] [--push]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from datasets import Dataset, DatasetDict, Video

SOURCE_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
TARGET_EPISODES = 1500
CANDIDATE_POOL = 1800
MIN_FRAMES, MAX_FRAMES = 15, 300  # 3-60 s at 5 fps
QUERY_CAM = "observation.images.image_1"
CORPUS_CAM = "observation.images.image_0"
SIZE_THRESHOLD = 20_000  # bytes; placeholder clips are ~2 KB
NUM_CHUNKS = 54  # ceil(53,192 / 1,000)


def video_path(episode_index: int, camera: str) -> str:
    return (
        f"videos/chunk-{episode_index // 1000:03d}/{camera}/"
        f"episode_{episode_index:06d}.mp4"
    )


def fetch(filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            SOURCE_REPO, filename, repo_type="dataset", cache_dir=str(cache_dir)
        )
    )


def real_video_episodes(api: HfApi, camera: str) -> set[int]:
    """Episode indices whose mp4 for `camera` is real (not a placeholder)."""
    episodes: set[int] = set()
    for chunk in range(NUM_CHUNKS):
        entries = api.list_repo_tree(
            SOURCE_REPO,
            path_in_repo=f"videos/chunk-{chunk:03d}/{camera}",
            repo_type="dataset",
            recursive=False,
        )
        for entry in entries:
            if getattr(entry, "size", 0) > SIZE_THRESHOLD:
                episodes.add(int(entry.path[-10:-4]))
    return episodes


def phase_sample(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    args.workdir.mkdir(parents=True, exist_ok=True)
    episodes_meta = fetch("meta/episodes.jsonl", cache_dir)

    api = HfApi()
    with ThreadPoolExecutor(max_workers=2) as pool:
        query_ok, corpus_ok = pool.map(
            lambda cam: real_video_episodes(api, cam), (QUERY_CAM, CORPUS_CAM)
        )
    both = query_ok & corpus_ok
    print(f"{len(both)} episodes have real videos for both cameras")

    seen_instructions: set[str] = set()
    unique: list[dict] = []
    with open(episodes_meta) as f:
        for line in f:
            ep = json.loads(line)
            instruction = ep["tasks"][0].strip()
            if (
                ep["episode_index"] not in both
                or not instruction
                or instruction.lower() in seen_instructions
                or not (MIN_FRAMES <= ep["length"] <= MAX_FRAMES)
            ):
                continue
            seen_instructions.add(instruction.lower())
            unique.append(
                {"episode_index": ep["episode_index"], "instruction": instruction}
            )
    step = max(1, len(unique) // CANDIDATE_POOL)
    candidates = unique[::step][:CANDIDATE_POOL]
    print(f"{len(candidates)} candidates after dedup/subsample; fetching videos")

    def grab_videos(cand: dict) -> dict | None:
        try:
            for cam in (QUERY_CAM, CORPUS_CAM):
                fetch(video_path(cand["episode_index"], cam), cache_dir)
            return cand
        except Exception as exc:  # noqa: BLE001 - skip broken episodes
            print(f"drop ep{cand['episode_index']}: {exc}")
            return None

    accepted: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(grab_videos, candidates):
            if result is not None:
                accepted.append(result)
            if len(accepted) >= TARGET_EPISODES:
                break
    accepted = sorted(accepted, key=lambda c: c["episode_index"])[:TARGET_EPISODES]
    print(f"{len(accepted)} episodes accepted")

    with open(args.workdir / "accepted.jsonl", "w") as f:
        for cand in accepted:
            f.write(json.dumps(cand) + "\n")
    print("sample phase done")


def phase_build(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    with open(args.workdir / "accepted.jsonl") as f:
        accepted = [json.loads(line) for line in f]

    query_rows, corpus_rows, qrel_rows = [], [], []
    for cand in accepted:
        idx = cand["episode_index"]
        query_rows.append(
            {
                "id": f"q-ep{idx:06d}",
                "video": str(fetch(video_path(idx, QUERY_CAM), cache_dir)),
            }
        )
        corpus_rows.append(
            {
                "id": f"c-ep{idx:06d}",
                "video": str(fetch(video_path(idx, CORPUS_CAM), cache_dir)),
            }
        )
        qrel_rows.append(
            {"query-id": f"q-ep{idx:06d}", "corpus-id": f"c-ep{idx:06d}", "score": 1}
        )

    queries = Dataset.from_list(query_rows).cast_column("video", Video())
    corpus = Dataset.from_list(corpus_rows).cast_column("video", Video())
    qrels = Dataset.from_list(qrel_rows)
    repo = f"{args.hf_user}/BridgeData-V2V"
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
