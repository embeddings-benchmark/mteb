"""Generate LaTeX table for MAEB benchmarks showing top 10 models.

This script reads results directly from JSON files to avoid environment issues.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add local mteb to path to use the development version
SCRIPT_DIR = Path(__file__).parent
MTEB_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MTEB_ROOT))

# Import benchmarks from mteb
from mteb.benchmarks import get_benchmark

MAEB_AUDIO = get_benchmark("MAEB(audio-only)")
MAEB_AUDIO_TEXT_LITE = get_benchmark("MAEB")


def get_task_names_from_benchmark(benchmark) -> list[str]:
    """Extract task names from a Benchmark object."""
    return [task.metadata.name for task in benchmark.tasks]


# Get task lists from benchmark definitions
MAEB_AUDIO_TASKS = get_task_names_from_benchmark(MAEB_AUDIO)
MAEB_AUDIO_TEXT_LITE_TASKS = get_task_names_from_benchmark(MAEB_AUDIO_TEXT_LITE)


def build_task_type_map(*benchmarks) -> dict[str, str]:
    """Build task type mapping from benchmark task metadata."""
    task_type_map = {}
    for benchmark in benchmarks:
        for task in benchmark.tasks:
            task_name = task.metadata.name
            task_type = task.metadata.type

            # For audio-text retrieval, distinguish A2T vs T2A based on task name
            if "A2TRetrieval" in task_name:
                task_type_map[task_name] = "A2TRetrieval"
            elif "T2ARetrieval" in task_name:
                task_type_map[task_name] = "T2ARetrieval"
            else:
                task_type_map[task_name] = task_type
    return task_type_map


# Build task type mapping dynamically from benchmarks
TASK_TYPE_MAP = build_task_type_map(MAEB_AUDIO, MAEB_AUDIO_TEXT_LITE)

# Task type abbreviations for the table
TASK_TYPE_ABBREV = {
    "AudioClassification": "Clf",
    "AudioMultilabelClassification": "Clf",  # Group with classification
    "AudioPairClassification": "PC",
    "AudioReranking": "Rrnk",
    "AudioClustering": "Clust",
    "AudioZeroshotClassification": "Zero Clf.",
    "Retrieval": "Rtrvl",
}

# Map task types to canonical categories for grouping
TASK_TYPE_CANONICAL = {
    "AudioClassification": "Classification",
    "AudioMultilabelClassification": "Classification",
    "AudioPairClassification": "PairClassification",
    "AudioReranking": "Reranking",
    "AudioClustering": "Clustering",
    "AudioZeroshotClassification": "ZeroshotClassification",
    "A2TRetrieval": "Retrieval",
    "T2ARetrieval": "Retrieval",
}


def load_results_from_json(results_dir: Path, task_names: list[str]) -> pd.DataFrame:
    """Load results directly from JSON files in the results directory."""
    # Structure: results_dir/model_name/revision/task_name.json
    results = {}

    for model_folder in results_dir.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name.replace("__", "/")

        for revision_folder in model_folder.iterdir():
            if not revision_folder.is_dir():
                continue

            revision = revision_folder.name
            model_key = f"{model_name}_{revision}"

            results[model_key] = {"model": model_name, "revision": revision}

            for task_file in revision_folder.glob("*.json"):
                task_name = task_file.stem

                if task_name not in task_names:
                    continue

                try:
                    with open(task_file) as f:
                        data = json.load(f)

                    # Extract main score from the result
                    # The structure varies, but typically scores are in 'scores' -> 'test' -> [{'main_score': X}]
                    if "scores" in data:
                        for split, split_data in data["scores"].items():
                            if split_data and isinstance(split_data, list):
                                for subset_data in split_data:
                                    if "main_score" in subset_data:
                                        # Use the main_score
                                        results[model_key][task_name] = subset_data[
                                            "main_score"
                                        ]
                                        break
                                break
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    continue

    # Convert to dataframe
    df = pd.DataFrame.from_dict(results, orient="index")
    return df


def compute_borda_count(df: pd.DataFrame, task_names: list[str]) -> pd.Series:
    """Compute Borda count for models based on task rankings."""
    # Get only task columns that exist
    available_tasks = [t for t in task_names if t in df.columns]

    borda_scores = pd.Series(0, index=df.index)

    for task in available_tasks:
        # Get valid scores for this task
        task_scores = df[task].dropna()
        if len(task_scores) == 0:
            continue

        # Rank models (higher score = better rank)
        ranks = task_scores.rank(ascending=False, method="min")

        # Borda count: (n - rank + 1) where n is number of models
        n = len(task_scores)
        borda_scores[task_scores.index] += n - ranks + 1

    return borda_scores


def compute_category_averages(
    df: pd.DataFrame, task_names: list[str]
) -> dict[str, pd.Series]:
    """Compute average scores per task category using canonical category names."""
    task_type_groups = defaultdict(list)

    for task_name in task_names:
        if task_name in df.columns:
            raw_type = TASK_TYPE_MAP.get(task_name, "Unknown")
            # Map to canonical category for grouping
            canonical_type = TASK_TYPE_CANONICAL.get(raw_type, raw_type)
            task_type_groups[canonical_type].append(task_name)

    category_avgs = {}
    for task_type, tasks in task_type_groups.items():
        available_tasks = [t for t in tasks if t in df.columns]
        if available_tasks:
            category_avgs[task_type] = df[available_tasks].mean(axis=1)

    return category_avgs


def find_best_scores(
    rankings_df: pd.DataFrame, task_types: list[str], category_avgs: dict
) -> dict:
    """Find best score for each column to highlight in bold."""
    best = {
        "mean": rankings_df["mean"].max(),
        "weighted_mean": rankings_df["weighted_mean"].max(),
    }
    for task_type in task_types:
        if task_type in category_avgs:
            best[task_type] = category_avgs[task_type].max()
    return best


def format_score(value: float, best_value: float, is_best: bool = None) -> str:
    """Format a score, bolding if it's the best."""
    if pd.isna(value):
        return "-"
    if is_best is None:
        is_best = abs(value - best_value) < 0.001
    if is_best:
        return f"\\textbf{{{value:.1f}}}"
    return f"{value:.1f}"


def generate_latex_table(
    benchmark_name: str,
    task_names: list[str],
    df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[str, list[str]]:
    """Generate LaTeX table rows for a benchmark."""
    # Filter to models that have at least one result
    available_tasks = [t for t in task_names if t in df.columns]
    df_filtered = df.dropna(subset=available_tasks, how="all")

    if len(df_filtered) == 0:
        return f"% No results for {benchmark_name}\n", []

    # Compute metrics
    borda_scores = compute_borda_count(df_filtered, task_names)
    category_avgs = compute_category_averages(df_filtered, task_names)

    # Compute mean and weighted mean
    df_filtered = df_filtered.copy()
    df_filtered["mean"] = df_filtered[available_tasks].mean(axis=1) * 100
    df_filtered["borda_count"] = borda_scores

    # Weighted mean (equal weight per category)
    unique_types = list(set(TASK_TYPE_MAP.get(t, "Unknown") for t in available_tasks))
    weighted_scores = pd.Series(0.0, index=df_filtered.index)
    valid_cat_counts = pd.Series(0, index=df_filtered.index)
    for task_type in unique_types:
        if task_type in category_avgs:
            cat_vals = category_avgs[task_type] * 100
            # Only add where we have valid values
            valid_mask = ~cat_vals.isna()
            weighted_scores[valid_mask] += cat_vals[valid_mask]
            valid_cat_counts[valid_mask] += 1

    # Avoid division by zero
    df_filtered["weighted_mean"] = weighted_scores / valid_cat_counts.replace(0, np.nan)

    # Sort by borda count
    df_filtered = df_filtered.sort_values("borda_count", ascending=False)

    # Get task type counts
    task_type_counts = defaultdict(int)
    for task_name in available_tasks:
        task_type = TASK_TYPE_MAP.get(task_name, "Unknown")
        task_type_counts[task_type] += 1

    # Sort task types for consistent ordering
    sorted_types = sorted(task_type_counts.keys())

    # Find best scores (among top N only)
    top_df = df_filtered.head(top_n)
    best_scores = {
        "mean": top_df["mean"].max(),
        "weighted_mean": top_df["weighted_mean"].max(),
    }
    for task_type in sorted_types:
        if task_type in category_avgs:
            top_indices = top_df.index
            cat_series = category_avgs[task_type]
            valid_vals = cat_series.loc[cat_series.index.intersection(top_indices)]
            if len(valid_vals) > 0:
                best_scores[task_type] = valid_vals.max() * 100

    # Build table rows
    lines = []

    # Benchmark section header
    total_tasks = len(available_tasks)
    type_counts = " & ".join(
        [f"\\textcolor{{gray}}{{({task_type_counts[t]})}}" for t in sorted_types]
    )

    lines.append(
        f"\\multicolumn{{{4 + len(sorted_types)}}}{{c}}{{\\vspace{{2mm}} \\normalsize \\texttt{{{benchmark_name}}}}} \\\\"
    )
    lines.append(
        f"\\textcolor{{gray}}{{Number of datasets}} & \\textcolor{{gray}}{{({total_tasks})}} & "
        f"\\textcolor{{gray}}{{({total_tasks})}} & \\textcolor{{gray}}{{({total_tasks})}} & {type_counts} \\\\"
    )
    lines.append("\\midrule")

    # Get top N models
    top_models = df_filtered.head(top_n)

    for rank, (idx, row) in enumerate(top_models.iterrows(), 1):
        model_name = row["model"]
        # Truncate long model names
        display_name = model_name.split("/")[-1] if "/" in model_name else model_name
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."

        borda = int(row["borda_count"])
        mean = row["mean"]
        weighted = row["weighted_mean"]

        # Format mean and weighted mean
        mean_str = format_score(mean, best_scores["mean"])
        weighted_str = format_score(weighted, best_scores["weighted_mean"])

        # Get category averages for this model
        cat_values = []
        for task_type in sorted_types:
            if task_type in category_avgs:
                try:
                    val = category_avgs[task_type].loc[idx] * 100
                    cat_values.append(
                        format_score(val, best_scores.get(task_type, val))
                    )
                except (KeyError, TypeError):
                    cat_values.append("-")
            else:
                cat_values.append("-")

        cat_str = " & ".join(cat_values)

        # Escape underscores in model name for LaTeX
        display_name = display_name.replace("_", "\\_")

        # Format the row
        lines.append(
            f"{display_name} & {rank} ({borda}) & {mean_str} & {weighted_str} & {cat_str} \\\\"
        )

    return "\n".join(lines), sorted_types


def generate_audio_table(
    df: pd.DataFrame, task_names: list[str], benchmark_name: str, top_n: int = 10
) -> str:
    """Generate LaTeX table for MAEB audio benchmark."""
    available_tasks = [t for t in task_names if t in df.columns]
    df_filtered = df.dropna(subset=available_tasks, how="all")

    if len(df_filtered) == 0:
        return f"% No results for {benchmark_name}\n"

    # Compute metrics
    borda_scores = compute_borda_count(df_filtered, task_names)
    category_avgs = compute_category_averages(df_filtered, task_names)

    df_filtered = df_filtered.copy()
    df_filtered["mean"] = df_filtered[available_tasks].mean(axis=1) * 100
    df_filtered["borda_count"] = borda_scores

    # Task type order - unified across all benchmarks for combined table
    task_type_order = [
        "Classification",
        "PairClassification",
        "Reranking",
        "Clustering",
        "Retrieval",
        "ZeroshotClassification",
    ]

    # Weighted mean using canonical categories (only categories that exist)
    weighted_scores = pd.Series(0.0, index=df_filtered.index)
    valid_cat_counts = pd.Series(0, index=df_filtered.index)
    for task_type in task_type_order:
        if task_type in category_avgs:
            cat_vals = category_avgs[task_type] * 100
            valid_mask = ~cat_vals.isna()
            weighted_scores[valid_mask] += cat_vals[valid_mask]
            valid_cat_counts[valid_mask] += 1
    df_filtered["weighted_mean"] = weighted_scores / valid_cat_counts.replace(0, np.nan)

    df_filtered = df_filtered.sort_values("borda_count", ascending=False)

    # Task type counts using canonical categories
    task_type_counts = defaultdict(int)
    for task_name in available_tasks:
        raw_type = TASK_TYPE_MAP.get(task_name, "Unknown")
        canonical_type = TASK_TYPE_CANONICAL.get(raw_type, raw_type)
        task_type_counts[canonical_type] += 1

    # Find best scores among top N
    top_df = df_filtered.head(top_n)
    best_scores = {
        "mean": top_df["mean"].max(),
        "weighted_mean": top_df["weighted_mean"].max(),
    }
    for task_type in task_type_order:
        if task_type in category_avgs:
            top_indices = top_df.index
            cat_series = category_avgs[task_type]
            valid_vals = cat_series.loc[cat_series.index.intersection(top_indices)]
            if len(valid_vals) > 0:
                best_scores[task_type] = valid_vals.max() * 100

    lines = []

    # Benchmark header - show all 6 categories for unified table
    total_tasks = len(available_tasks)
    type_counts = " & ".join(
        [
            f"\\textcolor{{gray}}{{({task_type_counts.get(t, 0)})}}"
            for t in task_type_order
        ]
    )

    lines.append(
        f"\\multicolumn{{10}}{{c}}{{\\vspace{{2mm}} \\normalsize \\texttt{{{benchmark_name}}}}} \\\\"
    )
    lines.append(
        f"\\textcolor{{gray}}{{Number of datasets}} & & \\textcolor{{gray}}{{({total_tasks})}} & "
        f"\\textcolor{{gray}}{{({total_tasks})}} & {type_counts} \\\\"
    )
    lines.append("\\midrule")

    # Model rows
    for rank, (idx, row) in enumerate(top_df.iterrows(), 1):
        model_name = row["model"]
        display_name = model_name.split("/")[-1] if "/" in model_name else model_name
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."

        borda = int(row["borda_count"])
        mean = row["mean"]
        weighted = row["weighted_mean"]

        mean_str = format_score(mean, best_scores["mean"])
        weighted_str = format_score(weighted, best_scores["weighted_mean"])

        cat_values = []
        for task_type in task_type_order:
            if task_type in category_avgs:
                try:
                    val = category_avgs[task_type].loc[idx] * 100
                    cat_values.append(
                        format_score(val, best_scores.get(task_type, val))
                    )
                except (KeyError, TypeError):
                    cat_values.append("-")
            else:
                cat_values.append("-")

        cat_str = " & ".join(cat_values)
        display_name = display_name.replace("_", "\\_")

        lines.append(
            f"{display_name} & {rank} ({borda}) & {mean_str} & {weighted_str} & {cat_str} \\\\"
        )

    return "\n".join(lines)


def generate_maeb_table_with_audio_rank(
    df: pd.DataFrame,
    maeb_tasks: list[str],
    audio_only_tasks: list[str],
    benchmark_name: str,
    top_n: int = 30,
) -> str:
    """Generate LaTeX table for MAEB benchmark with audio-only rank column."""
    available_tasks = [t for t in maeb_tasks if t in df.columns]
    df_filtered = df.dropna(subset=available_tasks, how="all")

    if len(df_filtered) == 0:
        return f"% No results for {benchmark_name}\n"

    # Compute MAEB metrics
    borda_scores = compute_borda_count(df_filtered, maeb_tasks)
    category_avgs = compute_category_averages(df_filtered, maeb_tasks)

    df_filtered = df_filtered.copy()
    df_filtered["mean"] = df_filtered[available_tasks].mean(axis=1) * 100
    df_filtered["borda_count"] = borda_scores

    # Compute audio-only rankings separately
    audio_available = [t for t in audio_only_tasks if t in df.columns]
    df_audio = df.dropna(subset=audio_available, how="all")
    audio_borda = compute_borda_count(df_audio, audio_only_tasks)
    # Create rank mapping for audio-only
    audio_rankings = audio_borda.sort_values(ascending=False)
    audio_rank_map = {idx: rank for rank, idx in enumerate(audio_rankings.index, 1)}

    # Task type order for MAEB
    task_type_order = [
        "Classification",
        "PairClassification",
        "Reranking",
        "Clustering",
        "Retrieval",
        "ZeroshotClassification",
    ]

    # Weighted mean using canonical categories
    weighted_scores = pd.Series(0.0, index=df_filtered.index)
    valid_cat_counts = pd.Series(0, index=df_filtered.index)
    for task_type in task_type_order:
        if task_type in category_avgs:
            cat_vals = category_avgs[task_type] * 100
            valid_mask = ~cat_vals.isna()
            weighted_scores[valid_mask] += cat_vals[valid_mask]
            valid_cat_counts[valid_mask] += 1
    df_filtered["weighted_mean"] = weighted_scores / valid_cat_counts.replace(0, np.nan)

    df_filtered = df_filtered.sort_values("borda_count", ascending=False)

    # Task type counts using canonical categories
    task_type_counts = defaultdict(int)
    for task_name in available_tasks:
        raw_type = TASK_TYPE_MAP.get(task_name, "Unknown")
        canonical_type = TASK_TYPE_CANONICAL.get(raw_type, raw_type)
        task_type_counts[canonical_type] += 1

    # Find best scores among top N
    top_df = df_filtered.head(top_n)
    best_scores = {
        "mean": top_df["mean"].max(),
        "weighted_mean": top_df["weighted_mean"].max(),
    }
    for task_type in task_type_order:
        if task_type in category_avgs:
            top_indices = top_df.index
            cat_series = category_avgs[task_type]
            valid_vals = cat_series.loc[cat_series.index.intersection(top_indices)]
            if len(valid_vals) > 0:
                best_scores[task_type] = valid_vals.max() * 100

    lines = []

    # Benchmark header
    total_tasks = len(available_tasks)
    type_counts = " & ".join(
        [
            f"\\textcolor{{gray}}{{({task_type_counts.get(t, 0)})}}"
            for t in task_type_order
        ]
    )

    lines.append(
        f"\\multicolumn{{11}}{{c}}{{\\vspace{{2mm}} \\normalsize \\texttt{{{benchmark_name}}}}} \\\\"
    )
    lines.append(
        f"\\textcolor{{gray}}{{Number of datasets}} & & & \\textcolor{{gray}}{{({total_tasks})}} & "
        f"\\textcolor{{gray}}{{({total_tasks})}} & {type_counts} \\\\"
    )
    lines.append("\\midrule")

    # Model rows
    for rank, (idx, row) in enumerate(top_df.iterrows(), 1):
        model_name = row["model"]
        display_name = model_name.split("/")[-1] if "/" in model_name else model_name
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."

        borda = int(row["borda_count"])
        mean = row["mean"]
        weighted = row["weighted_mean"]

        # Get audio-only rank for this model
        audio_rank = audio_rank_map.get(idx, "-")
        if audio_rank != "-":
            audio_rank_str = str(audio_rank)
        else:
            audio_rank_str = "-"

        mean_str = format_score(mean, best_scores["mean"])
        weighted_str = format_score(weighted, best_scores["weighted_mean"])

        cat_values = []
        for task_type in task_type_order:
            if task_type in category_avgs:
                try:
                    val = category_avgs[task_type].loc[idx] * 100
                    cat_values.append(
                        format_score(val, best_scores.get(task_type, val))
                    )
                except (KeyError, TypeError):
                    cat_values.append("-")
            else:
                cat_values.append("-")

        cat_str = " & ".join(cat_values)
        display_name = display_name.replace("_", "\\_")

        lines.append(
            f"{display_name} & {rank} & {audio_rank_str} & {mean_str} & {weighted_str} & {cat_str} \\\\"
        )

    return "\n".join(lines)


def main():
    results_dir = Path("/Users/isaac/work/maeb-results/results")

    # Load all results
    all_tasks = MAEB_AUDIO_TASKS + MAEB_AUDIO_TEXT_LITE_TASKS
    print(f"Loading results for {len(all_tasks)} tasks...")
    df = load_results_from_json(results_dir, all_tasks)
    print(f"Loaded results for {len(df)} models")

    print("Processing MAEB (top 30) with audio-only rank...")
    maeb_section = generate_maeb_table_with_audio_rank(
        df,
        maeb_tasks=MAEB_AUDIO_TEXT_LITE_TASKS,
        audio_only_tasks=MAEB_AUDIO_TASKS,
        benchmark_name="MAEB",
        top_n=30,
    )

    # Build single LaTeX table
    latex_output = []

    latex_output.append(r"""\begin{table*}[!th]
    \centering
    \caption{
    Top 30 models on the MAEB benchmark (35 tasks spanning audio-only and audio-text evaluation). Results are ranked using Borda count. The ``Audio'' column shows the model's rank on MAEB(audio-only) for reference. We provide averages across all tasks, and per task category. Task categories are abbreviated as: Classification (Clf), Pair Classification (PC), Reranking (Rrnk), Clustering (Clust), Retrieval (Rtrvl), Zero-shot Classification (Zero Clf.). We highlight the best score in \textbf{bold}.
    }
    \label{tab:maeb-performance}
    \resizebox{\textwidth}{!}{
    \setlength{\tabcolsep}{4pt}
    {\footnotesize
    \begin{tabular}{lcc|cc|cccccc}
    \toprule
    & \multicolumn{2}{c}{\textbf{Rank} ($\downarrow$)} & \multicolumn{2}{c}{\textbf{Average}} & \multicolumn{6}{c}{\textbf{Average per Category}} \\
    \cmidrule(r){2-3} \cmidrule{4-5} \cmidrule(l){6-11}
    \textbf{Model} & MAEB & Audio & All & Cat. & Clf & PC & Rrnk & Clust & Rtrvl & Zero Clf. \\
    \midrule""")

    latex_output.append(maeb_section)

    latex_output.append(r"""    \bottomrule
    \end{tabular}
    }
    } % end resize box
\end{table*}""")

    # Write to file
    output_path = Path("/Users/isaac/work/mteb/scripts/maeb_results_table.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(latex_output))

    print(f"\nLaTeX table written to {output_path}")


if __name__ == "__main__":
    main()
