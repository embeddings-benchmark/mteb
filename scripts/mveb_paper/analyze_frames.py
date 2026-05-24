import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
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

def plot_performance(df_to_plot: pd.DataFrame, title: str, ylabel: str, output_path: Path, ymin: float = None, legend_ncol: int = 1, overall_line: pd.Series = None, legend_fontsize: int = 14):
    """Reusable function to generate and save a line plot for performance scaling."""
    # Scale to 100 format
    df_to_plot = df_to_plot * 100
    if ymin is not None:
        ymin = ymin * 100

    if overall_line is not None:
        overall_line = overall_line * 100

    # Ensure index is string to make points equidistant
    df_to_plot.index = df_to_plot.index.astype(str)

    ax = df_to_plot.plot(kind='line', marker='o', figsize=(10, 6), fontsize=14)

    if overall_line is not None:
        ax.plot(range(len(overall_line)), overall_line.values, color='black', linestyle=':', linewidth=2, marker='o', label='Overall Performance')

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Number of Frames", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # Use categorical x-ticks
    ax.set_xticks(range(len(df_to_plot.index)))
    ax.set_xticklabels(df_to_plot.index, fontsize=14)

    ax.grid(True)

    # Adjust y-axis to make the scaling impact clear
    y_min_data, y_max_data = df_to_plot.min().min(), df_to_plot.max().max()
    
    if overall_line is not None:
        y_min_data = min(y_min_data, overall_line.min())
        y_max_data = max(y_max_data, overall_line.max())
        
    padding = (y_max_data - y_min_data) * 0.1
    if pd.notna(y_min_data) and pd.notna(y_max_data) and padding > 0:
        calculated_ymin = y_min_data - padding if ymin is None else ymin
        ax.set_ylim(calculated_ymin, y_max_data + padding)
    elif ymin is not None:
        ax.set_ylim(bottom=ymin)

    # Add legend at the bottom right
    ax.legend(loc='lower right', fontsize=legend_fontsize, ncol=legend_ncol)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure so subsequent plots don't overlap
    logging.info(f"Plot saved to {output_path}")

ALLOWED_MODELS = {
    "nvidia__omni-embed-nemotron-3b",
    "facebook__pe-av-small",
    "encord-team__ebind-full",
    "LCO-Embedding__LCO-Embedding-Omni-3B",
    "Haon-Chen__e5-omni-3B"
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
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to output diagrams/plots")
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
    if 64 in table1.columns:
        table1 = table1.sort_values(by=64, ascending=False)
    
    # Table 2: Model name as row x num_frames as col, mean performance across tasks
    table2 = df.groupby(["model_name", "num_frames"])["main_score"].mean().unstack()
    if 64 in table2.columns:
        table2 = table2.sort_values(by=64, ascending=False)

    # Table 3: Overall mean performance across all allowed tasks and models per num_frames
    table3 = df.groupby("num_frames")["main_score"].mean().to_frame("Overall Mean").T

    # Table 4: Task name as row x num_frames as col, mean performance across models
    table4 = df.groupby(["task_name", "num_frames"])["main_score"].mean().unstack()
    if 64 in table4.columns:
        table4 = table4.sort_values(by=64, ascending=False)

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
    
    print("\n" + "="*80)
    print("Table 4: Mean Performance by Task Name and Num Frames (Averaged across models)")
    print("="*80)
    print(table4.to_markdown(floatfmt=".4f"))

    # 6. Generate and save plots if requested
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot model performance scaling
        plot_performance(
            df_to_plot=table2.T,
            title="Effect of Frame Count on Model Performance",
            ylabel="Mean Performance",
            output_path=plot_dir / "model_performance_scaling.png"
        )

        # Plot overall performance scaling
        plot_performance(
            df_to_plot=table3.T,
            title="Overall Performance vs. Frame Count",
            ylabel="Overall Mean Performance",
            output_path=plot_dir / "overall_performance_scaling.png"
        )

        # Plot task performance scaling
        plot_performance(
            df_to_plot=table4.T,
            title="Effect of Frame Count Across Tasks",
            ylabel="Mean Performance",
            output_path=plot_dir / "task_performance_scaling.png",
            ymin=0.0,
            legend_ncol=2
        )

        # Plot model performance scaling with overall performance
        plot_performance(
            df_to_plot=table2.T,
            title="Effect of Frame Count on Model Performance (with Overall)",
            ylabel="Mean Performance",
            output_path=plot_dir / "model_performance_scaling_with_overall.png",
            overall_line=table3.loc["Overall Mean"],
            legend_fontsize=16
        )

if __name__ == "__main__":
    main()
