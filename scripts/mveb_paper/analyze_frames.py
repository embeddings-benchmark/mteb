import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import mteb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_task_type(task_name: str) -> str:
    """Attempt to retrieve the task type from MTEB metadata."""
    try:
        task = mteb.get_task(task_name)
        return task.metadata.type
    except Exception:
        return "Unknown"

ALLOWED_MODELS = {
    "nvidia__omni-embed-nemotron-3b",
    "facebook__pe-av-small",
    "encord-team__ebind-full"
}

ALLOWED_TASKS = {
    "MSRVTTT2V",
    "VATEXT2VARetrieval",
    "VATEXV2ARetrieval",
    "OmniVideoBenchVideoCentricQA",
    "WorldSense1MinVideoAudioCentricQA",
    "BreakfastClassification",
    "MusicAVQACLSVideoClustering"
}

def main():
    parser = argparse.ArgumentParser(description="Parse MTEB results across varying frame counts.")
    parser.add_argument("results_dir", type=str, help="Path to the root results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        logging.error(f"Directory '{results_dir}' does not exist.")
        return

    # Dictionary to keep track of uniquely processed task/model/frame combinations
    records = {}

    for json_path in results_dir.rglob("*.json"):
        # Ignore metadata JSON files
        if json_path.name == "model_meta.json":
            continue
            
        parts = json_path.parts
        
        # 1. Look for the `num_frame_X` folder level
        num_frame_idx = next((i for i, part in enumerate(parts) if part.startswith("num_frame_")), -1)
        
        # Ensure the path contains num_frames, model_name, revision, and the filename
        if num_frame_idx == -1 or num_frame_idx + 2 >= len(parts):
            continue
            
        try:
            num_frames = int(parts[num_frame_idx].split("_")[-1])
        except ValueError:
            continue
            
        model_name = parts[num_frame_idx + 1]
        task_name = json_path.stem
        
        if model_name not in ALLOWED_MODELS:
            continue
            
        if task_name not in ALLOWED_TASKS:
            continue

        # 2. Avoid duplicates: If task x model x num_frame is already recorded, skip
        key = (task_name, model_name, num_frames)
        if key in records:
            continue
            
        # 3. Read JSON and extract the score
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            test_scores = data.get("scores", {}).get("test", [])
            if not test_scores:
                continue
                
            # Average the main_score across subsets (if the task has multiple subsets)
            main_scores = [subset["main_score"] for subset in test_scores if "main_score" in subset]
                    
            if main_scores:
                avg_main_score = sum(main_scores) / len(main_scores)
                
                records[key] = {
                    "task_name": task_name,
                    "model_name": model_name,
                    "num_frames": num_frames,
                    "task_type": get_task_type(task_name),
                    "main_score": avg_main_score
                }
                logging.info(f"Using file: {json_path}")
        except Exception as e:
            logging.warning(f"Failed to process {json_path}: {e}")

    if not records:
        logging.info("No valid result files found. Please check your path.")
        return

    # 4. Load into Pandas DataFrame for easy aggregation
    df = pd.DataFrame(list(records.values()))

    # Table 1: Task type as row x num_frames as col, mean performance across models
    # `unstack()` pivots the `num_frames` from a row index to columns.
    table1 = df.groupby(["task_type", "num_frames"])["main_score"].mean().unstack()
    
    # Table 2: Model name as row x num_frames as col, mean performance across tasks
    table2 = df.groupby(["model_name", "num_frames"])["main_score"].mean().unstack()

    # Table 3: Overall mean performance across all allowed tasks and models per num_frames
    table3_mean = df.groupby("num_frames")["main_score"].mean().to_frame("Overall Mean").T
    table3_count = df.groupby("num_frames")["main_score"].count().to_frame("Entry Count").T
    table3 = pd.concat([table3_mean, table3_count])

    # 5. Output tables
    print("\n" + "="*80)
    print("Table 1: Mean Performance by Task Type and Num Frames (Averaged across models)")
    print("="*80)
    print(table1.to_markdown(floatfmt=".4f"))

    print("\n" + "="*80)
    print("Table 2: Mean Performance by Model and Num Frames (Averaged across tasks)")
    print("="*80)
    print(table2.to_markdown(floatfmt=".4f"))

    print("\n" + "="*80)
    print("Table 3: Overall Mean Performance by Num Frames")
    print("="*80)
    print(table3.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()
