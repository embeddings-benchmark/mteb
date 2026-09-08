"""Construction script for MetaWorld-MT50-I2V and MetaWorld-MT50-V2I.

Builds image<->video retrieval tasks from lerobot/metaworld_mt50:
- 49 robot manipulation tasks.
- For each task, 15 episodes are deterministically selected:
  - Lowest 5 indices become queries (49 * 5 = 245 queries).
  - Next 10 indices become corpus (49 * 10 = 490 corpus videos/images).
  - 0 leakage: query episodes are NEVER in the corpus pool.
- Each episode is rendered to an H.264 MP4 (256x256, 10 fps) via PyAV.
- The final frame of each episode is saved as the goal-state image.
- Both directions (I2V and V2I) are built and pushed to Hugging Face Hub:
  - iamfortytwo/MetaWorld-MT50-I2V
  - iamfortytwo/MetaWorld-MT50-V2I
"""

from __future__ import annotations

import argparse
import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Image, Video
from huggingface_hub import HfApi, create_repo, get_token, hf_hub_download
from PIL import Image as PILImage
from tqdm import tqdm

SOURCE_REPO = "lerobot/metaworld_mt50"
FPS = 10
QUERIES_PER_TASK = 5
CORPUS_PER_TASK = 10
TOTAL_PER_TASK = QUERIES_PER_TASK + CORPUS_PER_TASK
TARGET_SIZE = (256, 256)
CRF = 23


def encode_video_av(
    frames: list[bytes],
    out_path: Path,
    fps: int = FPS,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> None:
    container = av.open(str(out_path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = target_size[0]
    stream.height = target_size[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(CRF), "preset": "fast"}

    for f_bytes in frames:
        img = PILImage.open(io.BytesIO(f_bytes)).convert("RGB")
        if img.size != target_size:
            img = img.resize(target_size, PILImage.Resampling.BILINEAR)
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Meta-World MT50 I2V and V2I retrieval datasets."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.home() / ".cache" / "metaworld_retrieval",
        help="Local directory for temporary storage and media files.",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default="iamfortytwo",
        help="Hugging Face user or organization for repository upload.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=True,
        help="Push datasets to Hugging Face Hub (default: True).",
    )
    parser.add_argument(
        "--no-push",
        action="store_false",
        dest="push",
        help="Disable pushing to Hugging Face Hub.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel downloading and encoding.",
    )
    args = parser.parse_args()

    workdir: Path = args.workdir
    video_dir = workdir / "videos"
    image_dir = workdir / "goal_images"
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or get_token()

    print(f"Step 1: Downloading metadata from {SOURCE_REPO}...")
    episodes_meta_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset",
        token=token,
    )
    ep_tab = pq.read_table(
        episodes_meta_path,
        columns=["episode_index", "tasks", "data/file_index", "length"],
    )
    df = ep_tab.to_pandas()

    by_task: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        task_name = row["tasks"][0]
        by_task[task_name].append(
            {
                "episode_index": int(row["episode_index"]),
                "file_index": int(row["data/file_index"]),
                "length": int(row["length"]),
            }
        )

    print(f"Found {len(by_task)} tasks total.")

    # Select 15 episodes per task: 5 queries, 10 corpus
    query_pool: dict[str, list[int]] = {}
    corpus_pool: dict[str, list[int]] = {}
    selected_episodes: set[int] = set()
    file_to_episodes: dict[int, list[int]] = defaultdict(list)

    for task, eps in sorted(by_task.items()):
        eps.sort(key=lambda x: x["episode_index"])
        chosen = eps[:TOTAL_PER_TASK]
        q_eps = [e["episode_index"] for e in chosen[:QUERIES_PER_TASK]]
        c_eps = [e["episode_index"] for e in chosen[QUERIES_PER_TASK:TOTAL_PER_TASK]]
        query_pool[task] = q_eps
        corpus_pool[task] = c_eps
        for e in chosen:
            selected_episodes.add(e["episode_index"])
            file_to_episodes[e["file_index"]].append(e["episode_index"])

    total_queries = sum(len(v) for v in query_pool.values())
    total_corpus = sum(len(v) for v in corpus_pool.values())
    print(
        f"Selected {len(selected_episodes)} episodes: "
        f"{total_queries} queries, {total_corpus} corpus across {len(file_to_episodes)} parquet files."
    )

    # Render videos and goal images in parallel
    print(f"Step 2: Rendering MP4 videos and extracting goal images ({args.num_workers} workers)...")
    needed_files = sorted(file_to_episodes.keys())

    def process_file(f_idx: int) -> None:
        eps_in_file = file_to_episodes[f_idx]
        missing_eps = [
            ep
            for ep in eps_in_file
            if not (
                (video_dir / f"ep{ep:06d}.mp4").exists()
                and (image_dir / f"ep{ep:06d}.png").exists()
            )
        ]
        if not missing_eps:
            return

        parquet_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=f"data/chunk-000/file-{f_idx:03d}.parquet",
            repo_type="dataset",
            token=token,
        )
        tab = pq.read_table(
            parquet_path,
            columns=["episode_index", "observation.image"],
        )
        ep_indices = tab["episode_index"].to_pylist()
        img_rows = tab["observation.image"].to_pylist()

        frames_by_ep: dict[int, list[bytes]] = defaultdict(list)
        for ep, img_row in zip(ep_indices, img_rows):
            if ep in missing_eps:
                frames_by_ep[ep].append(img_row["bytes"])

        for ep in missing_eps:
            frames = frames_by_ep[ep]
            if not frames:
                continue
            mp4_path = video_dir / f"ep{ep:06d}.mp4"
            png_path = image_dir / f"ep{ep:06d}.png"

            encode_video_av(frames, mp4_path, fps=FPS, target_size=TARGET_SIZE)
            goal_img = PILImage.open(io.BytesIO(frames[-1])).convert("RGB")
            if goal_img.size != TARGET_SIZE:
                goal_img = goal_img.resize(TARGET_SIZE, PILImage.Resampling.BILINEAR)
            goal_img.save(png_path)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(
            tqdm(
                executor.map(process_file, needed_files),
                total=len(needed_files),
                desc="Processing parquet files",
            )
        )

    # Build HuggingFace datasets
    print("Step 3: Building Arrow datasets for I2V and V2I...")

    def build_direction(query_modality: str) -> dict[str, Dataset]:
        doc_modality = "video" if query_modality == "image" else "image"
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

        queries = Dataset.from_list(
            [{k: r[k] for k in ("id", query_modality)} for r in query_rows]
        ).cast_column(
            query_modality, Video() if query_modality == "video" else Image()
        )

        corpus = Dataset.from_list(
            [{k: r[k] for k in ("id", doc_modality)} for r in corpus_rows]
        ).cast_column(
            doc_modality, Video() if doc_modality == "video" else Image()
        )

        qrels = Dataset.from_list(qrel_rows)
        return {"queries": queries, "corpus": corpus, "qrels": qrels}

    api = HfApi(token=token)
    commit_hashes: dict[str, str] = {}

    for direction, query_modality in (("I2V", "image"), ("V2I", "video")):
        print(f"\nBuilding MetaWorld-MT50-{direction}...")
        parts = build_direction(query_modality)
        repo_id = f"{args.hf_user}/MetaWorld-MT50-{direction}"

        for config_name, ds in parts.items():
            print(f"  {repo_id} [{config_name}]: {len(ds)} rows")

        # Save locally to disk
        local_save_dir = workdir / f"MetaWorld-MT50-{direction}"
        DatasetDict({k: v for k, v in parts.items()}).save_to_disk(str(local_save_dir))
        print(f"  Saved locally to {local_save_dir}")

        if args.push:
            print(f"  Pushing {repo_id} to Hugging Face Hub...")
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    exist_ok=True,
                )
                for config_name, ds in parts.items():
                    print(f"  Uploading config '{config_name}'...")
                    DatasetDict({"test": ds}).push_to_hub(
                        repo_id,
                        config_name=config_name,
                        token=token,
                    )
                info = api.repo_info(repo_id=repo_id, repo_type="dataset")
                commit_hashes[direction] = info.sha
                print(f"  Successfully pushed {repo_id} at revision {info.sha}")
            except Exception as e:
                print(f"  WARNING: Push failed for {repo_id}: {e}")
                print("  Data is saved locally and can be pushed once write credentials are provided.")

    print("\nSummary:")
    for d, sha in commit_hashes.items():
        print(f"  MetaWorld-MT50-{d}: https://huggingface.co/datasets/{args.hf_user}/MetaWorld-MT50-{d} (commit {sha})")


if __name__ == "__main__":
    main()
