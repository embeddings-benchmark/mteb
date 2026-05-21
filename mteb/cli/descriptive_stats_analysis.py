import argparse
import json
from pathlib import Path


def get_test_nodes(node):
    """Recursively find all dictionaries under the key 'test'."""
    nodes = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "test" and isinstance(v, dict):
                nodes.append(v)
            else:
                nodes.extend(get_test_nodes(v))
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_avg_frames", type=float, default=None)
    parser.add_argument("--min_duration_seconds", type=float, default=None)
    parser.add_argument("--tasks", type=str, default="")
    args = parser.parse_args()

    target_tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else []

    # Assuming this script is at mteb/cli/descriptive_stats_analysis.py
    # Therefore, descriptive_stats is at mteb/descriptive_stats/
    base_dir = Path(__file__).resolve().parent.parent / "descriptive_stats"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
        
    target_keys = [
        "queries_video_statistics",
        "documents_video_statistics",
        "video_statistics",
        "video1_statistics",
        "video2_statistics",
    ]
    
    task_stats = {}

    for file_path in sorted(base_dir.rglob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        max_avg_frames = -1.0
        test_nodes = get_test_nodes(data)
        
        for test_node in test_nodes:
            for key in target_keys:
                if key in test_node and isinstance(test_node[key], dict):
                    stats = test_node[key]
                    avg_fps = stats.get("average_fps")
                    avg_duration = stats.get("average_duration_seconds")
                    
                    if avg_fps is not None and avg_duration is not None:
                        avg_frames = avg_fps * avg_duration
                        if avg_frames > max_avg_frames:
                            max_avg_frames = avg_frames
                            max_avg_duration = avg_duration
                            
        if max_avg_frames >= 0:
            task_stats[file_path.stem] = (max_avg_frames, max_avg_duration)

    group_above = []
    group_below = []
    group_rest = []

    for task_name, (avg_frames, avg_duration) in task_stats.items():
        if target_tasks and task_name in target_tasks:
            pass_frames = args.min_avg_frames is None or avg_frames > args.min_avg_frames
            pass_duration = args.min_duration_seconds is None or avg_duration > args.min_duration_seconds
            
            if pass_frames and pass_duration:
                group_above.append((task_name, avg_frames, avg_duration))
            else:
                group_below.append((task_name, avg_frames, avg_duration))
        else:
            group_rest.append((task_name, avg_frames, avg_duration))

    above_labels = []
    if args.min_avg_frames is not None:
        above_labels.append(f"> {args.min_avg_frames} avg frames")
    if args.min_duration_seconds is not None:
        above_labels.append(f"> {args.min_duration_seconds} avg duration")
    above_str = " and ".join(above_labels) if above_labels else "All Target Tasks"

    below_labels = []
    if args.min_avg_frames is not None:
        below_labels.append(f"<= {args.min_avg_frames} avg frames")
    if args.min_duration_seconds is not None:
        below_labels.append(f"<= {args.min_duration_seconds} avg duration")
    below_str = " or ".join(below_labels) if below_labels else "None"

    print(f"--- Passing minimum frames and duration threshold ({above_str}) ---")
    for task_name, avg_frames, avg_duration in group_above:
        print(f"Task: {task_name:<40} | Average Frames: {avg_frames:<8.2f} | Average Duration: {avg_duration:.2f}s")

    print(f"\n--- Not passing minimum frames and duration threshold ({below_str}) ---")
    for task_name, avg_frames, avg_duration in group_below:
        print(f"Task: {task_name:<40} | Average Frames: {avg_frames:<8.2f} | Average Duration: {avg_duration:.2f}s")

    print("\n--- Rest of the tasks ---")
    for task_name, avg_frames, avg_duration in group_rest:
        print(f"Task: {task_name:<40} | Average Frames: {avg_frames:<8.2f} | Average Duration: {avg_duration:.2f}s")


if __name__ == "__main__":
    main()