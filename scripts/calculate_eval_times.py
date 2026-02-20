#!/usr/bin/env python
"""Calculate total evaluation times for benchmarks and models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mteb.benchmarks import get_benchmark


def load_eval_times(revision_folder: Path, task_names: list[str]) -> dict[str, float]:
    """Load evaluation_time from task JSON files."""
    eval_times = {}
    for task_name in task_names:
        task_file = revision_folder / f"{task_name}.json"
        if task_file.exists():
            try:
                with open(task_file) as f:
                    data = json.load(f)
                if "evaluation_time" in data and data["evaluation_time"] is not None:
                    eval_times[task_name] = data["evaluation_time"]
            except (json.JSONDecodeError, KeyError):
                pass
    return eval_times


def main():
    parser = argparse.ArgumentParser(
        description="Calculate total evaluation times for benchmarks and models."
    )
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        required=True,
        help="List of benchmark names",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        required=True,
        help="List of model names (e.g., microsoft/msclap-2023)",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        default=None,
        help="Path to results directory (default: ~/.cache/mteb)",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir / "results"
    else:
        results_dir = Path.home() / ".cache" / "mteb" / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load benchmark task names
    benchmark_tasks = {}
    for bench_name in args.benchmarks:
        try:
            bench = get_benchmark(bench_name)
            benchmark_tasks[bench_name] = [task.metadata.name for task in bench.tasks]
        except Exception as e:
            print(f"Warning: Could not load benchmark '{bench_name}': {e}")
            continue

    if not benchmark_tasks:
        print("Error: No valid benchmarks found")
        return

    # Calculate eval times for each model and benchmark
    results = []
    for model_name in args.models:
        model_folder_name = model_name.replace("/", "__").replace(" ", "_")
        model_folder = results_dir / model_folder_name

        if not model_folder.exists():
            print(f"Warning: No results found for model '{model_name}'")
            continue

        # Find revision folder (use latest if multiple)
        revision_folders = [f for f in model_folder.iterdir() if f.is_dir()]
        if not revision_folders:
            print(f"Warning: No revision folders found for model '{model_name}'")
            continue

        # Sort by modification time, use latest
        revision_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        revision_folder = revision_folders[0]

        for bench_name, task_names in benchmark_tasks.items():
            eval_times = load_eval_times(revision_folder, task_names)
            total_time = sum(eval_times.values())
            total_hours = total_time / 3600

            results.append(
                {
                    "model": model_name,
                    "benchmark": bench_name,
                    "tasks_with_times": len(eval_times),
                    "total_tasks": len(task_names),
                    "total_seconds": total_time,
                    "total_hours": total_hours,
                }
            )

    # Print results
    if not results:
        print("No results found")
        return

    print("\n" + "=" * 80)
    print("Evaluation Time Summary")
    print("=" * 80)

    # Group by model
    current_model = None
    for r in results:
        if r["model"] != current_model:
            current_model = r["model"]
            print(f"\nModel: {current_model}")
            print("-" * 60)

        print(
            f"  {r['benchmark']:40s} "
            f"{r['tasks_with_times']:3d}/{r['total_tasks']:3d} tasks  "
            f"{r['total_hours']:8.3f} hours"
        )

    # Print totals per model
    print("\n" + "=" * 80)
    print("Totals by Model")
    print("=" * 80)

    model_totals = {}
    for r in results:
        if r["model"] not in model_totals:
            model_totals[r["model"]] = 0
        model_totals[r["model"]] += r["total_hours"]

    for model, total_hours in model_totals.items():
        print(f"  {model:50s} {total_hours:8.3f} hours")


if __name__ == "__main__":
    main()
