"""Construction script for RT1-T2V and RT1-V2T.

Builds text<->video retrieval tasks from
IPEC-COMMUNITY/fractal20220817_data_lerobot (LeRobot v2.0 conversion of
the RT-1 "fractal" data: 87,212 real-world Google Robot manipulation
episodes, one camera stream at 3 fps, 599 unique language instructions).

Construction:
- Episodes are filtered to 5-100 s. Instructions with at least
  EPISODES_PER_INSTruction eligible episodes are sorted alphabetically and
  evenly subsampled to NUM_INSTRUCTIONS (spreads skills: close/move/open/
  pick/place/...). For each kept instruction, EPISODES_PER_INSTRUCTION
  episodes are sampled evenly over its episode list.
- T2V: query = instruction text, corpus = all sampled videos; relevance is
  instruction-level and multi-positive (a query matches every corpus video
  of the same instruction).
- V2T: query = video (V2T_QUERIES_PER_INSTRUCTION per instruction, a
  subset of the same sampled episodes), corpus = every unique instruction
  in the source dataset (kept and unkept alike, so most corpus texts are
  distractors); each query has exactly one relevant text.

Usage:
    python build_rt1_datasets.py --workdir ~/MOEB/build/rt1 \
        --hf-user <username> [--phase sample|build] [--push]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

from datasets import Dataset, DatasetDict, Value, Video

SOURCE_REPO = "IPEC-COMMUNITY/fractal20220817_data_lerobot"
NUM_INSTRUCTIONS = 150
EPISODES_PER_INSTRUCTION = 10
V2T_QUERIES_PER_INSTRUCTION = 2
MIN_FRAMES, MAX_FRAMES = 15, 300  # 5-100 s at 3 fps
CAMERA = "observation.images.image"


def video_path(episode_index: int) -> str:
    return (
        f"videos/chunk-{episode_index // 1000:03d}/{CAMERA}/"
        f"episode_{episode_index:06d}.mp4"
    )


def fetch(filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            SOURCE_REPO, filename, repo_type="dataset", cache_dir=str(cache_dir)
        )
    )


def evenly(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def phase_sample(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    args.workdir.mkdir(parents=True, exist_ok=True)
    episodes_meta = fetch("meta/episodes.jsonl", cache_dir)

    by_instruction: dict[str, list[int]] = defaultdict(list)
    all_instructions: set[str] = set()
    with open(episodes_meta) as f:
        for line in f:
            ep = json.loads(line)
            instruction = ep["tasks"][0].strip()
            if not instruction:
                continue
            all_instructions.add(instruction)
            if MIN_FRAMES <= ep["length"] <= MAX_FRAMES:
                by_instruction[instruction].append(ep["episode_index"])

    eligible = sorted(
        instr
        for instr, eps in by_instruction.items()
        if len(eps) >= EPISODES_PER_INSTRUCTION
    )
    kept = evenly(eligible, NUM_INSTRUCTIONS)
    print(f"{len(eligible)} eligible instructions, keeping {len(kept)}")

    selection = {
        instr: evenly(sorted(by_instruction[instr]), EPISODES_PER_INSTRUCTION)
        for instr in kept
    }
    wanted = [(instr, idx) for instr, eps in selection.items() for idx in eps]
    print(f"fetching {len(wanted)} videos")

    def grab(item: tuple[str, int]) -> tuple[str, int] | None:
        try:
            fetch(video_path(item[1]), cache_dir)
            return item
        except Exception as exc:  # noqa: BLE001 - skip broken episodes
            print(f"drop ep{item[1]}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = [item for item in pool.map(grab, wanted) if item is not None]
    print(f"{len(accepted)} videos accepted")

    with open(args.workdir / "accepted.jsonl", "w") as f:
        for instr, idx in accepted:
            f.write(json.dumps({"instruction": instr, "episode_index": idx}) + "\n")
    with open(args.workdir / "all_instructions.json", "w") as f:
        json.dump(sorted(all_instructions), f)
    print("sample phase done")


def phase_build(args: argparse.Namespace) -> None:
    cache_dir = args.workdir / "hub_cache"
    with open(args.workdir / "accepted.jsonl") as f:
        accepted = [json.loads(line) for line in f]
    all_instructions = json.load(open(args.workdir / "all_instructions.json"))

    by_instruction: dict[str, list[int]] = defaultdict(list)
    for row in accepted:
        by_instruction[row["instruction"]].append(row["episode_index"])
    kept = sorted(by_instruction)
    instr_id = {instr: f"q-instr{i:03d}" for i, instr in enumerate(kept)}

    # T2V: instruction text -> all videos of that instruction.
    t2v_queries = Dataset.from_list(
        [{"id": instr_id[i], "text": i} for i in kept]
    ).cast_column("text", Value("string"))
    t2v_corpus = Dataset.from_list(
        [
            {"id": f"c-ep{idx:06d}", "video": str(fetch(video_path(idx), cache_dir))}
            for instr in kept
            for idx in sorted(by_instruction[instr])
        ]
    ).cast_column("video", Video())
    t2v_qrels = Dataset.from_list(
        [
            {"query-id": instr_id[instr], "corpus-id": f"c-ep{idx:06d}", "score": 1}
            for instr in kept
            for idx in sorted(by_instruction[instr])
        ]
    )

    # V2T: video -> the one matching instruction among ALL unique
    # instructions of the source dataset (mostly distractors).
    text_id = {text: f"c-text{i:03d}" for i, text in enumerate(all_instructions)}
    v2t_queries = Dataset.from_list(
        [
            {"id": f"q-ep{idx:06d}", "video": str(fetch(video_path(idx), cache_dir))}
            for instr in kept
            for idx in sorted(by_instruction[instr])[:V2T_QUERIES_PER_INSTRUCTION]
        ]
    ).cast_column("video", Video())
    v2t_corpus = Dataset.from_list(
        [{"id": text_id[t], "text": t} for t in all_instructions]
    ).cast_column("text", Value("string"))
    v2t_qrels = Dataset.from_list(
        [
            {"query-id": f"q-ep{idx:06d}", "corpus-id": text_id[instr], "score": 1}
            for instr in kept
            for idx in sorted(by_instruction[instr])[:V2T_QUERIES_PER_INSTRUCTION]
        ]
    )

    for direction, queries, corpus, qrels in (
        ("T2V", t2v_queries, t2v_corpus, t2v_qrels),
        ("V2T", v2t_queries, v2t_corpus, v2t_qrels),
    ):
        repo = f"{args.hf_user}/RT1-{direction}"
        for config_name, ds in (
            ("queries", queries),
            ("corpus", corpus),
            ("qrels", qrels),
        ):
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
