"""Construction script for YouCook2-I2V and YouCook2-V2I Retrieval Benchmarks.

Recipe-level non-leaking construction:
- Source: VLM2Vec/YouCook2 validation parquet and raw video clips.
- 84 recipes with >= 3 distinct video demonstrations.
- For each recipe:
  - 1 video demonstration is selected as the query.
    - Its final segment (plated dish) final frame is extracted as the goal dish image query.
    - Its first segment (cooking preparation) serves as the tutorial video for V2I.
  - 2-3 other video demonstrations (different cooks in different kitchens) form the corpus tutorial pool.
  - The query's own video is NEVER present in the corpus (0 frame leakage, 0 video leakage).
  - Relevance is recipe-level and multi-positive: a query matches instructional videos
    from different cooks making the same recipe (score 1). Distractor videos are recipes of other dishes.
- Outputs two symmetric benchmarks:
  - iamfortytwo/YouCook2-I2V (image -> video)
  - iamfortytwo/YouCook2-V2I (video -> image)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import av
from datasets import Dataset, DatasetDict, Features, Image, Value, Video
from huggingface_hub import HfApi, create_repo, get_token, hf_hub_download
import pyarrow.parquet as pq
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOURCE_REPO = "VLM2Vec/YouCook2"
SOURCE_PARQUET = "data/val-00000-of-00001.parquet"


def extract_final_frame(video_path: Path, output_image_path: Path) -> None:
    """Decode video with av and save its final frame as a JPEG."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    last_frame = None
    for frame in container.decode(stream):
        last_frame = frame
    container.close()

    if last_frame is None:
        raise RuntimeError(f"No video frames decoded from {video_path}")

    img = last_frame.to_image().convert("RGB")
    # Resize slightly if larger than 720p to save space while retaining visual fidelity
    max_dim = max(img.size)
    if max_dim > 720:
        scale = 720.0 / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, PILImage.Resampling.LANCZOS)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_image_path), format="JPEG", quality=90)


def build_youcook2_datasets(
    workdir: Path,
    hf_user: str = "iamfortytwo",
    token: str | None = None,
    push: bool = True,
    limit_recipes: int | None = None,
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    images_dir = workdir / "images"
    videos_dir = workdir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading metadata parquet from %s...", SOURCE_REPO)
    parquet_file = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=SOURCE_PARQUET,
        repo_type="dataset",
        token=token,
    )
    df = pq.read_table(parquet_file).to_pandas()
    logger.info("Loaded metadata with %d rows across %d unique recipes.", len(df), df["recipe_type"].nunique())

    api = HfApi(token=token)
    repo_files = set(api.list_repo_files(repo_id=SOURCE_REPO, repo_type="dataset"))

    # Group by recipe_type -> youtube_id -> list of segment rows
    from collections import defaultdict
    recipes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        recipes[str(row["recipe_type"])][str(row["youtube_id"])].append(row.to_dict())

    # Find recipes with >= 3 videos where each video has available files in raw_videos
    valid_recipes: dict[str, list[str]] = {}
    for r_type, yts in sorted(recipes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        valid_yts = []
        for yt, rows in yts.items():
            sorted_rows = sorted(rows, key=lambda x: int(x["id"].split("_")[-1]))
            first_id = sorted_rows[0]["id"]
            last_id = sorted_rows[-1]["id"]
            has_first = f"raw_videos/{first_id}.mp4" in repo_files or f"raw_videos/{first_id}.webm" in repo_files
            has_last = f"raw_videos/{last_id}.mp4" in repo_files or f"raw_videos/{last_id}.webm" in repo_files
            if has_first and has_last:
                valid_yts.append(yt)
        if len(valid_yts) >= 3:
            valid_recipes[r_type] = sorted(valid_yts)

    logger.info("Found %d recipes with >= 3 valid videos.", len(valid_recipes))
    if limit_recipes:
        valid_recipe_keys = sorted(valid_recipes.keys())[:limit_recipes]
        valid_recipes = {k: valid_recipes[k] for k in valid_recipe_keys}
        logger.info("Limiting to %d recipes for run.", len(valid_recipes))

    # Download required raw videos and extract frames in parallel
    download_tasks: list[tuple[str, str, Path]] = []
    extract_tasks: list[tuple[Path, Path]] = []

    # Map for each video: (image_path, video_path)
    video_artifacts: dict[str, tuple[Path, Path]] = {}

    for r_type, yts in valid_recipes.items():
        # Select 1 query and 2-3 corpus videos
        query_yt = yts[0]
        corpus_yts = yts[1:min(4, len(yts))]
        selected_yts = [query_yt] + corpus_yts

        for yt in selected_yts:
            if yt in video_artifacts:
                continue
            rows = sorted(recipes[r_type][yt], key=lambda x: int(x["id"].split("_")[-1]))
            first_row = rows[0]
            last_row = rows[-1]

            first_id = first_row["id"]
            last_id = last_row["id"]

            first_ext = "mp4" if f"raw_videos/{first_id}.mp4" in repo_files else "webm"
            last_ext = "mp4" if f"raw_videos/{last_id}.mp4" in repo_files else "webm"

            first_fn = f"raw_videos/{first_id}.{first_ext}"
            last_fn = f"raw_videos/{last_id}.{last_ext}"

            local_video_path = videos_dir / f"{yt}.{first_ext}"
            local_image_path = images_dir / f"{yt}.jpg"

            video_artifacts[yt] = (local_image_path, local_video_path)

            if not local_video_path.exists():
                download_tasks.append((first_fn, yt, local_video_path))

            # Need the last segment for frame extraction
            last_seg_path = workdir / "temp_segments" / f"{last_id}.{last_ext}"
            if not local_image_path.exists():
                if not last_seg_path.exists():
                    download_tasks.append((last_fn, yt, last_seg_path))
                extract_tasks.append((last_seg_path, local_image_path))

    logger.info("Downloading %d video clips...", len(download_tasks))

    def _download(item: tuple[str, str, Path]) -> None:
        fn, _, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=SOURCE_REPO, filename=fn, repo_type="dataset", token=token)
        shutil.copy2(cached, str(dst))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_download, download_tasks))

    logger.info("Extracting %d goal dish frames with PyAV...", len(extract_tasks))

    def _extract(item: tuple[Path, Path]) -> None:
        src_video, dst_image = item
        if not dst_image.exists():
            extract_final_frame(src_video, dst_image)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_extract, extract_tasks))

    # Clean up temporary segments
    temp_seg_dir = workdir / "temp_segments"
    if temp_seg_dir.exists():
        shutil.rmtree(str(temp_seg_dir), ignore_errors=True)

    # Build dataset rows
    query_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []

    for r_type in sorted(valid_recipes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        yts = valid_recipes[r_type]
        query_yt = yts[0]
        corpus_yts = yts[1:min(4, len(yts))]

        q_img, q_vid = video_artifacts[query_yt]
        qid = f"q-recipe-{r_type}-{query_yt}"
        query_rows.append({
            "id": qid,
            "image": str(q_img),
            "video": str(q_vid),
        })

        for c_yt in corpus_yts:
            c_img, c_vid = video_artifacts[c_yt]
            cid = f"c-recipe-{r_type}-{c_yt}"
            corpus_rows.append({
                "id": cid,
                "image": str(c_img),
                "video": str(c_vid),
            })
            qrel_rows.append({
                "query-id": qid,
                "corpus-id": cid,
                "score": 1,
            })

    logger.info(
        "Built dataset rows: queries=%d, corpus=%d, qrels=%d across %d recipes.",
        len(query_rows),
        len(corpus_rows),
        len(qrel_rows),
        len(valid_recipes),
    )

    def _make_benchmark(direction: str) -> dict[str, Dataset]:
        if direction == "I2V":
            q_mod, c_mod = "image", "video"
        else:
            q_mod, c_mod = "video", "image"

        q_ds = Dataset.from_list(
            [{k: r[k] for k in ("id", q_mod)} for r in query_rows]
        ).cast_column(q_mod, Video(decode=False) if q_mod == "video" else Image())

        c_ds = Dataset.from_list(
            [{k: r[k] for k in ("id", c_mod)} for r in corpus_rows]
        ).cast_column(c_mod, Video(decode=False) if c_mod == "video" else Image())

        qrels_features = Features({
            "query-id": Value("string"),
            "corpus-id": Value("string"),
            "score": Value("int32"),
        })
        qr_ds = Dataset.from_list(qrel_rows, features=qrels_features)

        return {"queries": q_ds, "corpus": c_ds, "qrels": qr_ds}

    commit_hashes: dict[str, str] = {}
    for direction in ("I2V", "V2I"):
        parts = _make_benchmark(direction)
        repo_id = f"{hf_user}/YouCook2-{direction}"
        logger.info("\nProcessing benchmark: %s", repo_id)
        for cfg, ds in parts.items():
            logger.info("  %s [%s]: %d rows", repo_id, cfg, len(ds))

        local_dir = workdir / f"YouCook2-{direction}"
        DatasetDict({k: v for k, v in parts.items()}).save_to_disk(str(local_dir))
        logger.info("  Saved locally to %s", local_dir)

        if push:
            logger.info("  Pushing to Hugging Face Hub under %s...", repo_id)
            try:
                create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
                for cfg, ds in parts.items():
                    logger.info("  Uploading %s [%s]...", repo_id, cfg)
                    DatasetDict({"test": ds}).push_to_hub(
                        repo_id,
                        config_name=cfg,
                        token=token,
                    )
                info = api.repo_info(repo_id=repo_id, repo_type="dataset")
                commit_hashes[direction] = info.sha
                logger.info("  Successfully pushed %s (commit: %s)", repo_id, info.sha)
            except Exception as e:
                logger.warning(
                    "  Push to Hugging Face Hub failed (%s). Data is saved locally in %s.",
                    e,
                    local_dir,
                )

    logger.info("\n=== Construction Complete ===")
    for d, sha in commit_hashes.items():
        logger.info("  YouCook2-%s: commit %s", d, sha)

    return commit_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create YouCook2 I2V and V2I retrieval datasets.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/youcook2_retrieval"),
        help="Local directory for video downloads, frame extraction, and local arrow datasets.",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default="iamfortytwo",
        help="Hugging Face user namespace.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN") or get_token(),
        help="Hugging Face token.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=True,
        help="Attempt to push datasets to Hugging Face Hub (default: True).",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Do not push datasets to Hugging Face Hub.",
    )
    parser.add_argument(
        "--limit-recipes",
        type=int,
        default=None,
        help="Optional limit on number of recipes for quick smoke testing.",
    )
    args = parser.parse_args()

    build_youcook2_datasets(
        workdir=args.workdir,
        hf_user=args.hf_user,
        token=args.token,
        push=args.push,
        limit_recipes=args.limit_recipes,
    )


if __name__ == "__main__":
    main()
