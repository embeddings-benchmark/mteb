"""Construction script for ManiSkill-I2V and ManiSkill-V2I.

Builds image<->video retrieval tasks from the official ManiSkill3
motion-planning demonstrations (haosulab/ManiSkill_Demonstrations): 8
tabletop manipulation tasks, EPISODES_PER_TASK successful episodes each,
replayed in simulation via environment states and rendered at 256x256.

Construction:
- Each episode is replayed with `set_state_dict` and rendered from TWO
  viewpoints: the environment's human render camera (full episode video)
  and the base sensor camera (final goal-state image only), so queries
  and documents never share a viewpoint.
- Per task, episodes are deterministically split into a query pool
  (QUERIES_PER_TASK episodes) and a corpus pool (the rest); a query's own
  source episode is never in the corpus. Relevance is task-level and
  multi-positive: a query matches every corpus episode of the same task.
  (Instance-level 1:1 designs over the near-duplicate corpus measured at
  chance level for current embedding models and were rejected.)
- I2V: query = goal image (base camera), corpus = videos (render camera).
  V2I: query = video, corpus = goal images.

Phases:
    render  -- replay + render all episodes (needs mani_skill installed;
               demos are downloaded automatically to ~/.maniskill/demos)
    build   -- package corpus/queries/qrels and optionally push to the Hub

Usage:
    python build_maniskill_datasets.py --workdir ~/MOEB/build/maniskill \
        --hf-user <username> --phase render [--tasks PickCube-v1 ...]
    python build_maniskill_datasets.py --workdir ~/MOEB/build/maniskill \
        --hf-user <username> --phase build --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

TASKS = [
    "DrawTriangle-v1",
    "PegInsertionSide-v1",
    "PickCube-v1",
    "PlugCharger-v1",
    "PullCubeTool-v1",
    "PushCube-v1",
    "StackCube-v1",
    "StackPyramid-v1",
]
EPISODES_PER_TASK = 150
QUERIES_PER_TASK = 10
SIZE = 256
VIDEO_FPS = 20  # matches the 20 Hz control frequency of the demos


def demo_traj_path(task: str) -> Path:
    return Path.home() / f".maniskill/demos/{task}/motionplanning/trajectory.h5"


def render_episode(env, traj_group, out_mp4: Path, out_png: Path) -> None:
    import numpy as np
    from PIL import Image as PILImage

    def state_at(i: int):
        def collect(group):
            return {
                k: (collect(v) if hasattr(v, "keys") else np.array(v[i]))
                for k, v in group.items()
            }

        return collect(traj_group["env_states"])

    n_steps = traj_group["actions"].shape[0] + 1
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
        f"{SIZE}x{SIZE}",
        "-r",
        str(VIDEO_FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n_steps):
        env.unwrapped.set_state_dict(state_at(i))
        frame = env.render()
        frame = (
            frame.squeeze().cpu().numpy()
            if hasattr(frame, "cpu")
            else np.asarray(frame).squeeze()
        )
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4}")

    # Goal-state image from the base sensor camera (different viewpoint).
    sensor = env.unwrapped.get_sensor_images()["base_camera"]["rgb"]
    goal = (
        sensor.squeeze().cpu().numpy()
        if hasattr(sensor, "cpu")
        else np.asarray(sensor).squeeze()
    )
    PILImage.fromarray(goal.astype(np.uint8)).save(out_png)


def phase_render(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401  (registers environments)
    from mani_skill.utils.download_demo import main as download_demo

    video_dir = args.workdir / "videos"
    image_dir = args.workdir / "goal_images"
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.workdir / f"manifest_{'_'.join(args.tasks)}.jsonl"
    manifest = open(manifest_path, "w")
    for task in args.tasks:
        traj_path = demo_traj_path(task)
        if not traj_path.exists():
            download_demo(argparse.Namespace(uid=task, output_dir=None, quiet=False))
        meta = json.load(open(str(traj_path).replace(".h5", ".json")))
        episodes = [e for e in meta["episodes"] if e.get("success", False)]
        episodes = episodes[: args.episodes_per_task]
        print(f"{task}: rendering {len(episodes)} successful episodes")

        env = gym.make(
            meta["env_info"]["env_id"],
            obs_mode="rgb",
            render_mode="rgb_array",
            sensor_configs=dict(width=SIZE, height=SIZE),
            human_render_camera_configs=dict(width=SIZE, height=SIZE),
        )
        with h5py.File(traj_path) as f:
            for ep in episodes:
                eid = ep["episode_id"]
                uid = f"{task}-ep{eid:04d}"
                mp4 = video_dir / f"{uid}.mp4"
                png = image_dir / f"{uid}.png"
                if not (mp4.exists() and png.exists()):
                    env.reset(
                        seed=ep["episode_seed"],
                        options=ep["reset_kwargs"].get("options", {}),
                    )
                    render_episode(env, f[f"traj_{eid}"], mp4, png)
                manifest.write(json.dumps({"uid": uid, "task": task}) + "\n")
        env.close()
    manifest.close()
    print("render phase done:", manifest_path)


def phase_build(args: argparse.Namespace) -> None:
    from collections import defaultdict

    from datasets import Dataset, DatasetDict, Image, Video

    video_dir = args.workdir / "videos"
    image_dir = args.workdir / "goal_images"
    by_task: dict[str, list[str]] = defaultdict(list)
    for manifest in args.workdir.glob("manifest_*.jsonl"):
        for line in open(manifest):
            row = json.loads(line)
            if row["uid"] not in by_task[row["task"]]:
                by_task[row["task"]].append(row["uid"])
    print(f"{sum(len(v) for v in by_task.values())} rendered episodes found")

    # Deterministic per-task split: lowest episode ids become queries.
    # Relevance is task-level and multi-positive; the query's own source
    # episode is never in the corpus. Near-duplicate 1:1 designs measured
    # at chance level for current models, see the PR discussion.
    query_pool: dict[str, list[str]] = {}
    corpus_pool: dict[str, list[str]] = {}
    for task, uids in sorted(by_task.items()):
        uids = sorted(uids)
        query_pool[task] = uids[:QUERIES_PER_TASK]
        corpus_pool[task] = uids[QUERIES_PER_TASK:]

    def path_of(uid: str, modality: str) -> str:
        if modality == "image":
            return str(image_dir / f"{uid}.png")
        return str(video_dir / f"{uid}.mp4")

    for direction, query_modality, doc_modality in (
        ("I2V", "image", "video"),
        ("V2I", "video", "image"),
    ):
        queries = Dataset.from_list(
            [
                {"id": f"q-{u}", query_modality: path_of(u, query_modality)}
                for t in sorted(by_task)
                for u in query_pool[t]
            ]
        ).cast_column(query_modality, Video() if query_modality == "video" else Image())
        corpus = Dataset.from_list(
            [
                {"id": f"c-{u}", doc_modality: path_of(u, doc_modality)}
                for t in sorted(by_task)
                for u in corpus_pool[t]
            ]
        ).cast_column(doc_modality, Video() if doc_modality == "video" else Image())
        qrels = Dataset.from_list(
            [
                {"query-id": f"q-{q}", "corpus-id": f"c-{c}", "score": 1}
                for t in sorted(by_task)
                for q in query_pool[t]
                for c in corpus_pool[t]
            ]
        )
        repo = f"{args.hf_user}/ManiSkill-{direction}"
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
    parser.add_argument("--phase", choices=["render", "build"], default="render")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes-per-task", type=int, default=EPISODES_PER_TASK)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.phase == "render":
        phase_render(args)
    else:
        phase_build(args)


if __name__ == "__main__":
    main()
