#!/usr/bin/env python3
"""Data preparation for WebVid-CoVR retrieval task in MTEB.

This script reads the test split metadata of 'lucas-ventura/WebVid-CoVR',
downloads the corresponding raw video files from 'lucas-ventura/WebVid',
extracts the middle frame of each reference video using PyAV, and constructs
three standardized retrieval splits:
- corpus: containing target videos.
- queries: containing composed queries with edit text instructions and middle frame PIL images.
- qrels: containing the relevance mapping between queries and corpus videos.

These are written to Hugging Face dataset deep9539/WebVid-CoVR.
"""

from __future__ import annotations

import io
import os
import av
import argparse
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, Image as HFImage, Video as HFVideo, Features, Value
from huggingface_hub import hf_hub_download


def extract_middle_frame(video_path: str | Path, output_image_path: str | Path) -> bool:
    """Extract the middle frame of a video file and save it as an image."""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        
        frames = []
        for frame in container.decode(video_stream):
            frames.append(frame.to_image())
            
        if not frames:
            # Fallback to black image
            img = Image.new("RGB", (224, 224), color="black")
            img.save(output_image_path)
            return False
            
        middle_frame = frames[len(frames) // 2]
        middle_frame.save(output_image_path)
        return True
    except Exception as e:
        print(f"Error decoding video {video_path}: {e}")
        # Fallback to black image on decoding error
        img = Image.new("RGB", (224, 224), color="black")
        img.save(output_image_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare WebVid-CoVR dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows processed.")
    parser.add_argument("--dry-run", action="store_true", help="Run locally without pushing to HF Hub.")
    parser.add_argument("--cache-dir", type=str, default="temp_webvid_covr", help="Temp folder for downloaded files.")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    videos_dir = cache_path / "videos"
    frames_dir = cache_path / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading test set metadata from lucas-ventura/WebVid-CoVR...")
    meta_path = hf_hub_download(
        repo_id="lucas-ventura/WebVid-CoVR",
        filename="webvid8m-covr_test.csv",
        repo_type="dataset"
    )
    df_meta = pd.read_csv(meta_path)
    if args.limit:
        df_meta = df_meta.head(args.limit)
    print(f"Loaded {len(df_meta)} metadata rows.")

    # Unique reference and target videos
    unique_ref_vids = df_meta["pth1"].unique().tolist()
    unique_tgt_vids = df_meta["pth2"].unique().tolist()
    all_unique_vids = list(set(unique_ref_vids + unique_tgt_vids))
    print(f"Total unique videos to download: {len(all_unique_vids)}")

    print("\nStep 2: Downloading video files from lucas-ventura/WebVid...")
    video_local_paths = {}
    
    def download_video(pth):
        filename = f"train/{pth}.mp4"
        try:
            local_file = hf_hub_download(
                repo_id="lucas-ventura/WebVid",
                filename=filename,
                repo_type="dataset"
            )
            # Create a symlink or copy to our videos cache folder to keep layout flat
            cached_dest = videos_dir / f"{pth.replace('/', '_')}.mp4"
            if not cached_dest.exists():
                shutil.copy(local_file, cached_dest)
            return pth, cached_dest
        except Exception as e:
            print(f"Error downloading video {pth}: {e}")
            return pth, None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_video, pth): pth for pth in all_unique_vids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            pth, dest_path = future.result()
            if dest_path:
                video_local_paths[pth] = dest_path

    print(f"Successfully downloaded {len(video_local_paths)} / {len(all_unique_vids)} videos.")

    print("\nStep 3: Extracting middle frames from reference videos...")
    frame_local_paths = {}
    
    def process_frame(pth):
        if pth not in video_local_paths:
            return pth, None
        out_path = frames_dir / f"{pth.replace('/', '_')}.jpg"
        success = extract_middle_frame(video_local_paths[pth], out_path)
        return pth, out_path if success else None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_frame, pth): pth for pth in unique_ref_vids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
            pth, out_path = future.result()
            if out_path:
                frame_local_paths[pth] = out_path

    print("\nStep 4: Building corpus, queries, and qrels datasets...")
    # 4a. Corpus: unique target videos (pth2)
    corpus_data = []
    for pth in unique_tgt_vids:
        if pth in video_local_paths:
            corpus_data.append({
                "id": pth,
                "video": str(video_local_paths[pth])
            })
    
    corpus_features = Features({
        "id": Value("string"),
        "video": HFVideo()
    })
    corpus_ds = Dataset.from_list(corpus_data, features=corpus_features)
    print(f"Corpus dataset has {len(corpus_ds)} candidate videos.")

    # 4b. Queries: Compositional queries (middle frame of pth1 + edit text)
    queries_data = []
    qrels_data = []
    
    for idx, row in df_meta.iterrows():
        pth1 = row["pth1"]
        pth2 = row["pth2"]
        edit_text = row["edit"]
        query_id = f"q-{idx}"
        
        if pth1 in frame_local_paths and pth2 in video_local_paths:
            queries_data.append({
                "id": query_id,
                "text": edit_text,
                "image": str(frame_local_paths[pth1])
            })
            qrels_data.append({
                "query-id": query_id,
                "corpus-id": pth2,
                "score": 1
            })

    queries_features = Features({
        "id": Value("string"),
        "text": Value("string"),
        "image": HFImage()
    })
    queries_ds = Dataset.from_list(queries_data, features=queries_features)
    
    qrels_features = Features({
        "query-id": Value("string"),
        "corpus-id": Value("string"),
        "score": Value("int32")
    })
    qrels_ds = Dataset.from_list(qrels_data, features=qrels_features)
    print(f"Queries dataset has {len(queries_ds)} entries.")
    print(f"Qrels dataset has {len(qrels_ds)} query-to-candidate relationships.")

    if args.dry_run:
        print("\n[Dry Run] Successfully prepared datasets locally. Skipping push to Hugging Face Hub.")
        print(f"Corpus dataset entries: {len(corpus_ds)}")
        print(f"Queries dataset entries: {len(queries_ds)}")
        print(f"Qrels dataset entries: {len(qrels_ds)}")
        return

    print("\nStep 5: Pushing prepared datasets to Hugging Face Hub (deep9539/WebVid-CoVR)...")
    
    print("Pushing corpus split...")
    corpus_ds.push_to_hub("deep9539/WebVid-CoVR", config_name="corpus", split="test")
    
    print("Pushing queries split...")
    queries_ds.push_to_hub("deep9539/WebVid-CoVR", config_name="queries", split="test")
    
    print("Pushing qrels split...")
    qrels_ds.push_to_hub("deep9539/WebVid-CoVR", config_name="qrels", split="test")
    
    # Optional clean-up
    print("\nCleaning up local temporary cache folder...")
    shutil.rmtree(cache_path, ignore_errors=True)
    
    print("\nSuccessfully finished data preparation and upload to Hugging Face Hub!")


if __name__ == "__main__":
    main()
