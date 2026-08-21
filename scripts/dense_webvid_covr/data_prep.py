#!/usr/bin/env python3
"""Data preparation for Dense-WebVid-CoVR retrieval task in MTEB.

This script reads the test split metadata of 'omkarthawakar/Dense-WebVid-CoVR',
downloads the corresponding raw video files, and constructs
three standardized retrieval splits:
- corpus: containing target videos.
- queries: containing composed queries with edit text instructions and reference videos.
- qrels: containing the relevance mapping between queries and corpus videos.

These are written to Hugging Face dataset nik1995/Dense-WebVid-CoVR.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Video as HFVideo, Features, Value
from huggingface_hub import hf_hub_download, login


def main():
    parser = argparse.ArgumentParser(description="Prepare Dense-WebVid-CoVR dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows processed (useful for testing).")
    parser.add_argument("--dry-run", action="store_true", help="Run locally without pushing to HF Hub.")
    parser.add_argument("--cache-dir", type=str, default="temp_dense_webvid_covr", help="Temp folder for downloaded files.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token to authenticate.")
    args = parser.parse_args()

    if args.token:
        print("Logging in to Hugging Face...")
        login(token=args.token)

    cache_path = Path(args.cache_dir)
    videos_dir = cache_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading test set metadata from omkarthawakar/Dense-WebVid-CoVR...")
    meta_path = hf_hub_download(
        repo_id="omkarthawakar/Dense-WebVid-CoVR",
        filename="annotations/dense-webvid8m-covr_test.csv",
        repo_type="dataset"
    )
    df_meta = pd.read_csv(meta_path)
    print(f"Loaded {len(df_meta)} metadata rows.")

    if args.limit:
        df_meta = df_meta.head(args.limit)
        print(f"Limited metadata to {len(df_meta)} rows for testing.")

    # Unique reference (pth1) and target (pth2) videos
    unique_ref_vids = df_meta["pth1"].unique().tolist()
    unique_tgt_vids = df_meta["pth2"].unique().tolist()
    all_unique_vids = list(set(unique_ref_vids + unique_tgt_vids))
    print(f"Total unique videos to download: {len(all_unique_vids)}")

    print("\nStep 2: Downloading video files from omkarthawakar/Dense-WebVid-CoVR...")
    video_local_paths = {}
    
    def download_video(pth):
        filename = f"WebVid/8M/train/{pth}.mp4"
        try:
            local_file = hf_hub_download(
                repo_id="omkarthawakar/Dense-WebVid-CoVR",
                filename=filename,
                repo_type="dataset"
            )
            # Create a flat cached destination
            cached_dest = videos_dir / f"{pth.replace('/', '_')}.mp4"
            if not cached_dest.exists():
                shutil.copy(local_file, cached_dest)
            return pth, cached_dest
        except Exception as e:
            print(f"Error downloading video {pth}: {e}")
            return pth, None

    # Use parallel threads for extremely fast downloading
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_video, pth): pth for pth in all_unique_vids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            pth, dest_path = future.result()
            if dest_path:
                video_local_paths[pth] = dest_path

    print(f"Successfully downloaded {len(video_local_paths)} / {len(all_unique_vids)} videos.")

    print("\nStep 3: Building corpus, queries, and qrels datasets...")
    
    # 3a. Corpus: unique target videos (pth2) that were successfully downloaded
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

    # 3b. Queries: Compositional queries (reference video of pth1 + edit text)
    queries_data = []
    qrels_data = []
    
    for idx, row in df_meta.iterrows():
        pth1 = row["pth1"]
        pth2 = row["pth2"]
        edit_text = row["edit"]
        query_id = f"q-{row['index']}"
        
        if pth1 in video_local_paths and pth2 in video_local_paths:
            queries_data.append({
                "id": query_id,
                "text": edit_text,
                "video": str(video_local_paths[pth1])
            })
            qrels_data.append({
                "query-id": query_id,
                "corpus-id": pth2,
                "score": 1
            })

    queries_features = Features({
        "id": Value("string"),
        "text": Value("string"),
        "video": HFVideo()
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

    print("\nStep 4: Pushing prepared datasets to Hugging Face Hub (nik1995/Dense-WebVid-CoVR)...")
    
    print("Pushing corpus split...")
    corpus_ds.push_to_hub("nik1995/Dense-WebVid-CoVR", config_name="corpus", split="test")
    
    print("Pushing queries split...")
    queries_ds.push_to_hub("nik1995/Dense-WebVid-CoVR", config_name="queries", split="test")
    
    print("Pushing qrels split...")
    qrels_ds.push_to_hub("nik1995/Dense-WebVid-CoVR", config_name="qrels", split="test")
    
    # Optional clean-up
    print("\nCleaning up local temporary cache folder...")
    shutil.rmtree(cache_path, ignore_errors=True)
    
    print("\nSuccessfully finished data preparation and upload to Hugging Face Hub!")


if __name__ == "__main__":
    main()
