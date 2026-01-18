#!/usr/bin/env python3
"""
Analyze MAEB task selection to identify redundancies and optimize coverage.

This script:
1. Loads task metadata from MAEB(extended)
2. Groups tasks by source dataset to identify same-source task families
3. Computes task-to-task correlation matrix using model results
4. Analyzes language/domain/category coverage
5. Recommends task removals based on redundancy and coverage preservation

Usage:
    python scripts/analyze_maeb_task_selection.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import mteb
from mteb.cache import ResultCache


# Models for evaluation time calculation
EVAL_TIME_MODELS = [
    ("yamnet", "google/yamnet"),
    ("xls-r-2b", "facebook/wav2vec2-xls-r-2b"),
    ("clap_gen", "laion/larger_clap_general"),
    ("htsat", "laion/clap-htsat-fused"),
]

# Manually protected tasks (user preferences)
MANUALLY_PROTECTED_TASKS = [
    "BirdCLEF",  # Prefer over BirdSet for Bioacoustics domain
]

# Tasks to exclude (redundant with manually protected tasks)
TASKS_TO_EXCLUDE = [
    "BirdSet",  # Redundant with BirdCLEF (same Bioacoustics domain)
]

# Retrieval task families - for each family, prefer T2A over A2T
# If both A2TRetrieval and T2ARetrieval exist, remove A2T
RETRIEVAL_FAMILIES = [
    "Fleurs",
    "GigaSpeech",
    "HiFiTTS",
    "LibriTTS",
    "JamAltLyric",
    "AudioCaps",
    "AudioSetStrong",
    "CMUArctic",
    "Clotho",
    "CommonVoice17",
    "CommonVoice21",
    "EmoVDB",
    "JLCorpus",
    "MACS",
    "MusicCaps",
    "SoundDescs",
    "UrbanSound8K",
    "GoogleSVQ",
]

# Same-source families for deduplication
# Tasks from the same family and same task type are considered redundant
SAME_SOURCE_FAMILIES = [
    "CommonLanguage",  # Age, Gender, Language detection
    "FSD",  # FSD2019Kaggle, FSD50K, FSDnoisy18k
    "IEMOCAP",  # Emotion, Gender
    "VoxPopuli",  # Various ID and clustering tasks
    "ESC50",  # Multiple task types
    "GTZAN",  # Genre classification, clustering, reranking
    "UrbanSound8K",  # Multiple task types (but different types OK)
    "CREMA",  # Multiple task types
    "VocalSound",  # Multiple task types
]


def deduplicate_retrieval_directions(
    task_names: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Remove A2T retrieval tasks when T2A exists for the same family.

    For retrieval tasks, T2A (text-to-audio) is generally preferred over A2T
    (audio-to-text) as it represents a more common use case.

    Args:
        task_names: List of task names

    Returns:
        Tuple of (filtered task names, list of (removed_task, reason) tuples)
    """
    removed = []
    remaining = []

    # Build sets of A2T and T2A tasks by family
    a2t_tasks = {}  # family -> task name
    t2a_tasks = {}  # family -> task name

    for task in task_names:
        for family in RETRIEVAL_FAMILIES:
            if task.startswith(family):
                if "A2TRetrieval" in task:
                    a2t_tasks[family] = task
                elif "T2ARetrieval" in task:
                    t2a_tasks[family] = task
                break

    # Remove A2T if T2A exists for same family
    tasks_to_remove = set()
    for family, a2t_task in a2t_tasks.items():
        if family in t2a_tasks:
            tasks_to_remove.add(a2t_task)
            removed.append(
                (
                    a2t_task,
                    f"Prefer T2A over A2T for {family} family (keeping {t2a_tasks[family]})",
                )
            )

    for task in task_names:
        if task not in tasks_to_remove:
            remaining.append(task)

    return remaining, removed


def enforce_retrieval_direction_preference(
    task_names: list[str],
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Post-processing step: Remove A2T retrieval tasks when T2A exists for the same family.

    This is applied after correlation-based selection to ensure we never keep
    both A2T and T2A for the same retrieval family.

    Args:
        task_names: List of selected task names

    Returns:
        Tuple of (filtered task names, list of (removed_task, reason, 0.0) tuples)
    """
    remaining, removed_pairs = deduplicate_retrieval_directions(task_names)
    # Convert to the expected format with correlation value
    removed = [(task, reason, 0.0) for task, reason in removed_pairs]
    return remaining, removed


def deduplicate_same_source_families(
    task_names: list[str],
    results_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Remove same-type tasks from the same source family, keeping the one with lowest correlation.

    For each family with multiple tasks of the same type, keeps only one task
    (the one with lowest average correlation to other retained tasks).

    Args:
        task_names: List of selected task names
        results_df: Model results DataFrame for correlation calculation
        metadata_df: Task metadata DataFrame

    Returns:
        Tuple of (filtered task names, list of (removed_task, reason, 0.0) tuples)
    """
    remaining = task_names.copy()
    removed = []

    # Build task -> (family, type) mapping
    task_to_family = {}
    for task in task_names:
        for family in SAME_SOURCE_FAMILIES:
            if task.startswith(family) or family.lower() in task.lower():
                task_rows = metadata_df[metadata_df["name"] == task]
                if len(task_rows) > 0:
                    task_meta = task_rows.iloc[0]
                    task_to_family[task] = (family, task_meta["type"])
                break

    # Group tasks by (family, type)
    family_type_groups = defaultdict(list)
    for task, (family, task_type) in task_to_family.items():
        family_type_groups[(family, task_type)].append(task)

    # For groups with >1 task, keep only the one with lowest avg correlation
    for (family, task_type), tasks in family_type_groups.items():
        if len(tasks) <= 1:
            continue

        # Compute average correlation for each task with all other retained tasks
        best_task = None
        best_avg_corr = float("inf")

        for candidate in tasks:
            if candidate not in results_df.columns:
                continue

            other_tasks = [
                t for t in remaining if t != candidate and t in results_df.columns
            ]
            if not other_tasks:
                continue

            correlations = []
            for other in other_tasks:
                corr = results_df[[candidate, other]].corr(method="spearman").iloc[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            if correlations:
                avg_corr = np.mean(correlations)
                if avg_corr < best_avg_corr:
                    best_avg_corr = avg_corr
                    best_task = candidate

        # Remove all but the best task
        if best_task:
            for task in tasks:
                if task != best_task and task in remaining:
                    remaining.remove(task)
                    removed.append(
                        (
                            task,
                            f"Same-family ({family}) same-type ({task_type}) redundancy, keeping {best_task}",
                            0.0,
                        )
                    )

    return remaining, removed


def load_eval_times(
    results_dir: Path,
    model_name: str,
    task_names: list[str],
) -> dict[str, float]:
    """Load evaluation_time from task JSON files for a model.

    Args:
        results_dir: Path to results directory (e.g., ~/.cache/mteb/results)
        model_name: Model name (e.g., "laion/larger_clap_music_and_speech")
        task_names: List of task names to load times for

    Returns:
        Dictionary mapping task name to evaluation time in seconds
    """
    model_folder_name = model_name.replace("/", "__").replace(" ", "_")
    model_folder = results_dir / model_folder_name

    if not model_folder.exists():
        return {}

    # Find revision folder (use latest if multiple)
    revision_folders = [f for f in model_folder.iterdir() if f.is_dir()]
    if not revision_folders:
        return {}

    # Sort by modification time, use latest
    revision_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    revision_folder = revision_folders[0]

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


def compute_total_eval_time(
    eval_times: dict[str, float],
    task_names: list[str],
) -> tuple[float, int]:
    """Compute total evaluation time for a set of tasks.

    Args:
        eval_times: Dictionary mapping task name to evaluation time
        task_names: List of task names to sum

    Returns:
        Tuple of (total_seconds, tasks_with_times)
    """
    total = 0.0
    count = 0
    for task in task_names:
        if task in eval_times:
            total += eval_times[task]
            count += 1
    return total, count


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1h 23m" or "45m 30s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


def get_task_metadata_df(benchmark_name: str) -> pd.DataFrame:
    """Extract task metadata into a DataFrame.

    Args:
        benchmark_name: Name of the benchmark to analyze

    Returns:
        DataFrame with columns: name, languages, domains, type, category, dataset_path
    """
    benchmark = mteb.get_benchmark(benchmark_name)

    rows = []
    for task in benchmark.tasks:
        meta = task.metadata
        rows.append(
            {
                "name": meta.name,
                "languages": tuple(sorted(meta.languages)) if meta.languages else (),
                "domains": tuple(sorted(meta.domains)) if meta.domains else (),
                "type": meta.type,
                "category": meta.category,
                "dataset_path": meta.dataset.get("path", ""),
                "modalities": tuple(sorted(meta.modalities)),
            }
        )

    return pd.DataFrame(rows)


def get_same_source_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group tasks by source dataset.

    Args:
        df: DataFrame with task metadata (must have 'name' and 'dataset_path' columns)

    Returns:
        Dictionary mapping dataset path to list of task names
    """
    groups = defaultdict(list)
    for _, row in df.iterrows():
        dataset_path = row["dataset_path"]
        # Normalize dataset path - extract the base dataset name
        base_path = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
        # Remove common suffixes that indicate variants
        for suffix in ["-retrieval", "-classification", "-clustering"]:
            if base_path.lower().endswith(suffix):
                base_path = base_path[: -len(suffix)]
        groups[row["dataset_path"]].append(row["name"])

    # Only return groups with multiple tasks
    return {path: tasks for path, tasks in groups.items() if len(tasks) > 1}


def identify_task_families(df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify task families based on shared naming patterns.

    Tasks like VoxPopuliAccentID, VoxPopuliAccentClustering, VoxPopuliGenderID
    all share "VoxPopuli" as a common prefix.

    Args:
        df: DataFrame with task metadata

    Returns:
        Dictionary mapping family name to list of task names
    """
    families = defaultdict(list)
    task_names = df["name"].tolist()

    # Common dataset prefixes to look for
    common_prefixes = [
        "VoxPopuli",
        "VoxCeleb",
        "ESC50",
        "UrbanSound8K",
        "GTZAN",
        "FSD",
        "VocalSound",
        "CREMA",
        "CommonLanguage",
        "IEMOCAP",
        "AudioCaps",
        "MusicCaps",
        "LibriTTS",
        "GigaSpeech",
        "Fleurs",
        "CommonVoice",
        "SpokenSQuAD",
        "JamAlt",
        "Clotho",
        "AudioSet",
    ]

    for task_name in task_names:
        matched = False
        for prefix in common_prefixes:
            if task_name.startswith(prefix) or prefix.lower() in task_name.lower():
                families[prefix].append(task_name)
                matched = True
                break
        if not matched:
            families["Other"].append(task_name)

    # Only return families with multiple tasks
    return {
        family: sorted(tasks) for family, tasks in families.items() if len(tasks) > 1
    }


def load_model_results(results_dir: Path | str) -> tuple[pd.DataFrame, list[str]]:
    """Load model results from ResultCache.

    Args:
        results_dir: Path to results directory

    Returns:
        Tuple of (results DataFrame, list of task names with results)
    """
    results_dir = Path(results_dir)
    model_names = [
        folder.name.replace("__", "/")
        for folder in results_dir.iterdir()
        if folder.is_dir()
    ]
    print(f"Found {len(model_names)} models")

    # Get model metadata
    models = [mteb.get_model_meta(name) for name in model_names]

    # Load all audio tasks
    audio_tasks = mteb.get_tasks(modalities=["audio"])

    # Load results
    cache = ResultCache(cache_path=str(results_dir.parent))
    mteb_results = cache.load_results(
        models=models, tasks=audio_tasks, require_model_meta=False
    )

    # Create DataFrame
    full_df = mteb_results.to_dataframe().set_index("task_name").T

    # Filter models with too many NaN values
    nan_counts = full_df.isna().sum(axis=1)
    results_df = full_df[nan_counts <= 10]

    print(f"Models after NaN filter (<=10 NaN): {len(results_df)}")
    print(f"Tasks with results: {len(results_df.columns)}")

    return results_df, list(results_df.columns)


def compute_task_correlation(
    results_df: pd.DataFrame, tasks: list[str]
) -> pd.DataFrame:
    """Compute pairwise Spearman correlation matrix between tasks.

    Args:
        results_df: DataFrame with model results (rows=models, cols=tasks)
        tasks: List of task names to compute correlations for

    Returns:
        Correlation matrix as DataFrame
    """
    task_df = results_df[tasks].select_dtypes(include=["number"])
    return task_df.corr(method="spearman")


def compute_benchmark_correlation(
    results_df: pd.DataFrame,
    source_tasks: list[str],
    selected_tasks: list[str],
) -> tuple[float, float]:
    """Compute correlation between source benchmark and selected subset.

    Args:
        results_df: DataFrame with model results (rows=models, cols=tasks)
        source_tasks: List of source benchmark task names
        selected_tasks: List of selected task names (subset of source)

    Returns:
        Tuple of (spearman_correlation, pearson_correlation)
    """
    # Filter to available tasks
    source_available = [t for t in source_tasks if t in results_df.columns]
    selected_available = [t for t in selected_tasks if t in results_df.columns]

    if not source_available or not selected_available:
        return float("nan"), float("nan")

    # Compute average performance per model
    source_avg = results_df[source_available].mean(axis=1)
    selected_avg = results_df[selected_available].mean(axis=1)

    # Drop any NaN values
    valid_mask = ~(source_avg.isna() | selected_avg.isna())
    source_avg = source_avg[valid_mask]
    selected_avg = selected_avg[valid_mask]

    if len(source_avg) < 3:
        return float("nan"), float("nan")

    spearman_corr, _ = spearmanr(source_avg, selected_avg)
    pearson_corr, _ = pearsonr(source_avg, selected_avg)

    return spearman_corr, pearson_corr


def get_coverage_analysis(task_names: list[str]) -> dict:
    """Analyze language/domain/category/type coverage for a set of tasks.

    Args:
        task_names: List of task names to analyze

    Returns:
        Dictionary with coverage statistics
    """
    tasks = mteb.get_tasks(tasks=task_names)

    languages = set()
    domains = set()
    categories = set()
    types = set()
    type_counts = Counter()
    category_counts = Counter()
    domain_counts = Counter()

    for task in tasks:
        meta = task.metadata
        if meta.languages:
            languages.update(meta.languages)
        if meta.domains:
            domains.update(meta.domains)
            domain_counts.update(meta.domains)
        if meta.category:
            categories.add(meta.category)
            category_counts[meta.category] += 1
        types.add(meta.type)
        type_counts[meta.type] += 1

    return {
        "n_tasks": len(task_names),
        "n_languages": len(languages),
        "n_domains": len(domains),
        "n_categories": len(categories),
        "n_types": len(types),
        "languages": sorted(languages),
        "domains": sorted(domains),
        "categories": sorted(categories),
        "types": sorted(types),
        "type_counts": dict(type_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "domain_counts": dict(domain_counts.most_common()),
    }


def get_highly_correlated_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Get all task pairs with correlation above threshold.

    Args:
        corr_matrix: Correlation matrix
        threshold: Minimum correlation threshold

    Returns:
        List of (task1, task2, correlation) tuples, sorted by correlation descending
    """
    pairs = []
    cols = corr_matrix.columns.tolist()

    for i, task1 in enumerate(cols):
        for task2 in cols[i + 1 :]:
            corr_val = corr_matrix.loc[task1, task2]
            if not np.isnan(corr_val) and corr_val > threshold:
                pairs.append((task1, task2, corr_val))

    return sorted(pairs, key=lambda x: x[2], reverse=True)


def get_unique_coverage_tasks(
    task_names: list[str],
) -> dict[str, list[str]]:
    """Identify tasks that provide unique language/domain/category coverage.

    Args:
        task_names: List of task names

    Returns:
        Dictionary with lists of tasks that are unique for each dimension
    """
    tasks = mteb.get_tasks(tasks=task_names)

    # Build reverse mappings
    lang_to_tasks = defaultdict(list)
    domain_to_tasks = defaultdict(list)
    category_to_tasks = defaultdict(list)
    type_to_tasks = defaultdict(list)

    for task in tasks:
        meta = task.metadata
        name = meta.name
        if meta.languages:
            for lang in meta.languages:
                lang_to_tasks[lang].append(name)
        if meta.domains:
            for domain in meta.domains:
                domain_to_tasks[domain].append(name)
        if meta.category:
            category_to_tasks[meta.category].append(name)
        type_to_tasks[meta.type].append(name)

    # Find tasks with unique coverage
    unique_lang_tasks = []
    unique_domain_tasks = []
    unique_category_tasks = []
    unique_type_tasks = []

    for lang, task_list in lang_to_tasks.items():
        if len(task_list) == 1:
            unique_lang_tasks.append((task_list[0], lang))

    for domain, task_list in domain_to_tasks.items():
        if len(task_list) == 1:
            unique_domain_tasks.append((task_list[0], domain))

    for category, task_list in category_to_tasks.items():
        if len(task_list) == 1:
            unique_category_tasks.append((task_list[0], category))

    for task_type, task_list in type_to_tasks.items():
        if len(task_list) == 1:
            unique_type_tasks.append((task_list[0], task_type))

    return {
        "unique_language": unique_lang_tasks,
        "unique_domain": unique_domain_tasks,
        "unique_category": unique_category_tasks,
        "unique_type": unique_type_tasks,
    }


def recommend_removals(
    corr_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    threshold: float = 0.8,
    protected_tasks: list[str] | None = None,
) -> list[dict]:
    """Recommend tasks to remove based on correlation and redundancy analysis.

    Args:
        corr_matrix: Pairwise task correlation matrix
        metadata_df: DataFrame with task metadata
        threshold: Correlation threshold for considering pairs redundant
        protected_tasks: Tasks that should not be removed (e.g., provide unique coverage)

    Returns:
        List of removal recommendations with justification
    """
    if protected_tasks is None:
        protected_tasks = []

    recommendations = []
    current_tasks = list(corr_matrix.columns)
    task_families = identify_task_families(metadata_df)

    # Get highly correlated pairs
    high_corr_pairs = get_highly_correlated_pairs(corr_matrix, threshold)

    for task1, task2, corr_val in high_corr_pairs:
        if task1 not in current_tasks or task2 not in current_tasks:
            continue

        # Determine which task to potentially remove
        # Preference: keep t2a over a2t for retrieval, keep classification over clustering

        task1_meta = metadata_df[metadata_df["name"] == task1].iloc[0]
        task2_meta = metadata_df[metadata_df["name"] == task2].iloc[0]

        # Check if either is protected
        if task1 in protected_tasks and task2 in protected_tasks:
            recommendations.append(
                {
                    "pair": (task1, task2),
                    "correlation": corr_val,
                    "action": "keep_both",
                    "reason": "Both tasks provide unique coverage",
                }
            )
            continue

        # Score each task for removal (higher = more likely to remove)
        def removal_score(task_name: str, meta: pd.Series) -> float:
            score = 0
            # Prefer removing a2t over t2a for retrieval tasks
            if meta["category"] == "a2t":
                score += 1
            elif meta["category"] == "t2a":
                score -= 1
            # Prefer keeping classification over clustering of same source
            if "Clustering" in task_name:
                score += 0.5
            if "Classification" in task_name or "ID" in task_name:
                score -= 0.5
            # Penalize if protected
            if task_name in protected_tasks:
                score -= 10
            return score

        score1 = removal_score(task1, task1_meta)
        score2 = removal_score(task2, task2_meta)

        if score1 > score2:
            remove, keep = task1, task2
        else:
            remove, keep = task2, task1

        # Check if in same family
        in_same_family = False
        for family, members in task_families.items():
            if task1 in members and task2 in members:
                in_same_family = True
                break

        recommendations.append(
            {
                "pair": (task1, task2),
                "correlation": corr_val,
                "recommend_remove": remove,
                "recommend_keep": keep,
                "same_family": in_same_family,
                "reason": f"High correlation ({corr_val:.3f})"
                + (", same source dataset" if in_same_family else ""),
            }
        )

    return recommendations


def analyze_maeb_vs_extended(maeb_tasks: list[str], extended_tasks: list[str]) -> dict:
    """Compare MAEB selection against MAEB(extended).

    Args:
        maeb_tasks: List of task names in MAEB
        extended_tasks: List of task names in MAEB(extended)

    Returns:
        Comparison analysis
    """
    maeb_coverage = get_coverage_analysis(maeb_tasks)
    extended_coverage = get_coverage_analysis(extended_tasks)

    # Languages lost
    maeb_langs = set(maeb_coverage["languages"])
    extended_langs = set(extended_coverage["languages"])
    lost_langs = extended_langs - maeb_langs

    # Domains lost
    maeb_domains = set(maeb_coverage["domains"])
    extended_domains = set(extended_coverage["domains"])
    lost_domains = extended_domains - maeb_domains

    # Categories lost
    maeb_cats = set(maeb_coverage["categories"])
    extended_cats = set(extended_coverage["categories"])
    lost_cats = extended_cats - maeb_cats

    # Types lost
    maeb_types = set(maeb_coverage["types"])
    extended_types = set(extended_coverage["types"])
    lost_types = extended_types - maeb_types

    return {
        "maeb_coverage": maeb_coverage,
        "extended_coverage": extended_coverage,
        "lost_languages": sorted(lost_langs),
        "lost_domains": sorted(lost_domains),
        "lost_categories": sorted(lost_cats),
        "lost_types": sorted(lost_types),
        "compression_ratio": len(maeb_tasks) / len(extended_tasks),
    }


def is_removal_valid(
    current_tasks: list[str],
    task_to_remove: str,
    protected_tasks: set[str],
) -> bool:
    """Check if removing a task would lose unique coverage.

    Args:
        current_tasks: Current list of tasks
        task_to_remove: Task being considered for removal
        protected_tasks: Tasks that must not be removed

    Returns:
        True if removal is valid (won't lose unique coverage)
    """
    if task_to_remove in protected_tasks:
        return False

    remaining_tasks = [t for t in current_tasks if t != task_to_remove]
    if not remaining_tasks:
        return False

    task = mteb.get_task(task_to_remove)
    remaining = mteb.get_tasks(tasks=remaining_tasks)

    # Check unique task type
    remaining_types = {t.metadata.type for t in remaining}
    if task.metadata.type not in remaining_types:
        return False

    # Check unique category
    if task.metadata.category:
        remaining_cats = {t.metadata.category for t in remaining if t.metadata.category}
        if task.metadata.category not in remaining_cats:
            return False

    # Check unique domain
    if task.metadata.domains:
        remaining_domains = set()
        for t in remaining:
            if t.metadata.domains:
                remaining_domains.update(t.metadata.domains)
        for domain in task.metadata.domains:
            if domain not in remaining_domains:
                return False

    # Check unique language
    if task.metadata.languages:
        remaining_langs = set()
        for t in remaining:
            if t.metadata.languages:
                remaining_langs.update(t.metadata.languages)
        for lang in task.metadata.languages:
            if lang not in remaining_langs:
                return False

    return True


def iterative_task_selection(
    results_df: pd.DataFrame,
    initial_tasks: list[str],
    metadata_df: pd.DataFrame,
    threshold: float = 0.8,
    protected_tasks: set[str] | None = None,
    prefer_remove_same_source: bool = True,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Iteratively remove highly correlated tasks while preserving coverage.

    Args:
        results_df: Model results DataFrame
        initial_tasks: Starting list of tasks
        metadata_df: Task metadata DataFrame
        threshold: Correlation threshold for removal
        protected_tasks: Tasks that must not be removed
        prefer_remove_same_source: Prefer removing tasks from same source family

    Returns:
        Tuple of (remaining tasks, list of removed (task, reason, correlation))
    """
    if protected_tasks is None:
        protected_tasks = set()

    current_tasks = initial_tasks.copy()
    removed_tasks = []
    task_families = identify_task_families(metadata_df)

    # Build reverse mapping: task -> family
    task_to_family = {}
    for family, members in task_families.items():
        for member in members:
            task_to_family[member] = family

    while True:
        # Recompute correlation for remaining tasks
        available = [t for t in current_tasks if t in results_df.columns]
        if len(available) < 2:
            break

        corr_matrix = compute_task_correlation(results_df, available)
        pairs = get_highly_correlated_pairs(corr_matrix, threshold)

        if not pairs:
            break

        removed_this_round = False

        for task1, task2, corr_val in pairs:
            if task1 not in current_tasks or task2 not in current_tasks:
                continue

            # Get metadata
            meta1 = metadata_df[metadata_df["name"] == task1].iloc[0]
            meta2 = metadata_df[metadata_df["name"] == task2].iloc[0]

            # Score tasks for removal (higher = prefer to remove)
            def removal_priority(task_name: str, meta: pd.Series) -> float:
                score = 0

                # RULE: For retrieval families, strongly prefer removing A2T when T2A exists
                # And strongly protect T2A when A2T exists (so we keep T2A, not A2T)
                if "A2TRetrieval" in task_name:
                    for family in RETRIEVAL_FAMILIES:
                        if task_name.startswith(family):
                            # Check if T2A exists for this family in current tasks
                            t2a_task = f"{family}T2ARetrieval"
                            if t2a_task in current_tasks:
                                score += 100  # Very high priority to remove A2T when T2A exists
                            break
                elif "T2ARetrieval" in task_name:
                    for family in RETRIEVAL_FAMILIES:
                        if task_name.startswith(family):
                            # Check if A2T exists for this family in current tasks
                            a2t_task = f"{family}A2TRetrieval"
                            if a2t_task in current_tasks:
                                score -= 100  # Strongly protect T2A when A2T exists
                            break

                # Prefer removing a2t over t2a for retrieval (general preference)
                if meta["category"] == "a2t":
                    score += 2
                elif meta["category"] == "t2a":
                    score -= 1

                # Prefer removing clustering over classification
                if "Clustering" in task_name:
                    score += 1
                if "Classification" in task_name or "ID" in task_name:
                    score -= 0.5

                # Count how many tasks from same family are still present
                family = task_to_family.get(task_name)
                if family and prefer_remove_same_source:
                    family_count = sum(
                        1 for t in current_tasks if task_to_family.get(t) == family
                    )
                    if family_count > 1:
                        score += (
                            family_count  # More redundant = higher removal priority
                        )

                return score

            score1 = removal_priority(task1, meta1)
            score2 = removal_priority(task2, meta2)

            # Try to remove the higher-scored task first
            candidates = sorted(
                [(task1, score1, meta1), (task2, score2, meta2)],
                key=lambda x: x[1],
                reverse=True,
            )

            for task_name, _, meta in candidates:
                if is_removal_valid(current_tasks, task_name, protected_tasks):
                    current_tasks.remove(task_name)
                    family = task_to_family.get(task_name, "")
                    reason = f"corr={corr_val:.3f} with {task1 if task_name == task2 else task2}"
                    if family:
                        reason += f", family={family}"
                    removed_tasks.append((task_name, reason, corr_val))
                    removed_this_round = True
                    break

            if removed_this_round:
                break

        if not removed_this_round:
            # No more removable pairs above threshold
            break

    return current_tasks, removed_tasks


def print_report(
    metadata_df: pd.DataFrame,
    corr_matrix: pd.DataFrame | None,
    maeb_tasks: list[str],
    extended_tasks: list[str],
):
    """Print comprehensive analysis report.

    Args:
        metadata_df: Task metadata DataFrame
        corr_matrix: Correlation matrix (or None if not available)
        maeb_tasks: Current MAEB task list
        extended_tasks: MAEB(extended) task list
    """
    print("=" * 80)
    print("MAEB Task Selection Analysis Report")
    print("=" * 80)

    # Task family analysis
    print("\n## Task Families (Same Source Dataset)")
    print("-" * 60)
    families = identify_task_families(metadata_df)
    for family, tasks in sorted(families.items(), key=lambda x: -len(x[1])):
        print(f"\n### {family} ({len(tasks)} tasks)")
        for task in tasks:
            in_maeb = "✓" if task in maeb_tasks else "✗"
            task_meta = metadata_df[metadata_df["name"] == task].iloc[0]
            print(
                f"  [{in_maeb}] {task} ({task_meta['type']}, {task_meta['category']})"
            )

    # VoxPopuli-specific analysis
    print("\n## VoxPopuli Task Analysis")
    print("-" * 60)
    vox_tasks = [t for t in extended_tasks if "VoxPopuli" in t]
    vox_in_maeb = [t for t in vox_tasks if t in maeb_tasks]
    print(f"VoxPopuli tasks in MAEB(extended): {len(vox_tasks)}")
    print(f"VoxPopuli tasks in MAEB: {len(vox_in_maeb)}")
    for task in vox_tasks:
        in_maeb = "✓ MAEB" if task in maeb_tasks else "  ext"
        task_meta = metadata_df[metadata_df["name"] == task].iloc[0]
        print(f"  [{in_maeb}] {task}")
        print(
            f"           Type: {task_meta['type']}, Category: {task_meta['category']}"
        )

    # Coverage comparison
    print("\n## Coverage Comparison: MAEB vs MAEB(extended)")
    print("-" * 60)
    comparison = analyze_maeb_vs_extended(maeb_tasks, extended_tasks)

    print(f"\nTask counts: MAEB={len(maeb_tasks)}, Extended={len(extended_tasks)}")
    print(f"Compression ratio: {comparison['compression_ratio']:.1%}")

    print(
        f"\nLanguages: MAEB={comparison['maeb_coverage']['n_languages']}, Extended={comparison['extended_coverage']['n_languages']}"
    )
    if comparison["lost_languages"]:
        print(f"  Languages lost: {len(comparison['lost_languages'])}")
        print(
            f"  {comparison['lost_languages'][:10]}{'...' if len(comparison['lost_languages']) > 10 else ''}"
        )

    print(
        f"\nDomains: MAEB={comparison['maeb_coverage']['n_domains']}, Extended={comparison['extended_coverage']['n_domains']}"
    )
    if comparison["lost_domains"]:
        print(f"  Domains lost: {comparison['lost_domains']}")

    print(
        f"\nCategories: MAEB={comparison['maeb_coverage']['n_categories']}, Extended={comparison['extended_coverage']['n_categories']}"
    )
    if comparison["lost_categories"]:
        print(f"  Categories lost: {comparison['lost_categories']}")

    print(
        f"\nTask types: MAEB={comparison['maeb_coverage']['n_types']}, Extended={comparison['extended_coverage']['n_types']}"
    )
    if comparison["lost_types"]:
        print(f"  Types lost: {comparison['lost_types']}")

    # Type distribution
    print("\n## Task Type Distribution")
    print("-" * 60)
    print("\nMAEB:")
    for t, count in comparison["maeb_coverage"]["type_counts"].items():
        print(f"  {t}: {count}")
    print("\nMAEB(extended):")
    for t, count in comparison["extended_coverage"]["type_counts"].items():
        print(f"  {t}: {count}")

    # Category distribution
    print("\n## Category Distribution")
    print("-" * 60)
    print("\nMAEB:")
    for cat, count in comparison["maeb_coverage"]["category_counts"].items():
        print(f"  {cat}: {count}")
    print("\nMAEB(extended):")
    for cat, count in comparison["extended_coverage"]["category_counts"].items():
        print(f"  {cat}: {count}")

    # Correlation analysis (if available)
    if corr_matrix is not None:
        print("\n## Highly Correlated Task Pairs (threshold=0.8)")
        print("-" * 60)

        # Filter to tasks in current MAEB
        maeb_corr = corr_matrix.loc[
            [t for t in maeb_tasks if t in corr_matrix.index],
            [t for t in maeb_tasks if t in corr_matrix.columns],
        ]

        high_corr = get_highly_correlated_pairs(maeb_corr, threshold=0.8)
        if high_corr:
            print(f"\nFound {len(high_corr)} highly correlated pairs in MAEB:")
            for task1, task2, corr_val in high_corr[:20]:
                task1_meta = metadata_df[metadata_df["name"] == task1].iloc[0]
                task2_meta = metadata_df[metadata_df["name"] == task2].iloc[0]
                print(f"\n  {task1} <-> {task2}: {corr_val:.3f}")
                print(f"    {task1}: {task1_meta['type']}, {task1_meta['category']}")
                print(f"    {task2}: {task2_meta['type']}, {task2_meta['category']}")
        else:
            print("\nNo pairs above threshold 0.8 in current MAEB selection.")

        # Recommendations
        print("\n## Removal Recommendations")
        print("-" * 60)

        # Get unique coverage tasks
        unique_tasks = get_unique_coverage_tasks(maeb_tasks)
        protected = set()
        for dim, task_list in unique_tasks.items():
            for task, coverage in task_list:
                protected.add(task)

        print(f"\nProtected tasks (unique coverage): {len(protected)}")
        for dim, task_list in unique_tasks.items():
            if task_list:
                print(f"  {dim}:")
                for task, coverage in task_list:
                    print(f"    - {task} (unique: {coverage})")

        recommendations = recommend_removals(
            maeb_corr, metadata_df, threshold=0.8, protected_tasks=list(protected)
        )

        if recommendations:
            print(f"\nRecommendations for {len(recommendations)} pairs:")
            for rec in recommendations:
                print(f"\n  Pair: {rec['pair'][0]} <-> {rec['pair'][1]}")
                print(f"  Correlation: {rec['correlation']:.3f}")
                if "recommend_remove" in rec:
                    print(f"  Recommend remove: {rec['recommend_remove']}")
                    print(f"  Recommend keep: {rec['recommend_keep']}")
                print(f"  Reason: {rec['reason']}")

    # Unique coverage analysis for MAEB(extended) not in MAEB
    print("\n## Tasks in MAEB(extended) but not MAEB with Unique Coverage")
    print("-" * 60)
    excluded_tasks = [t for t in extended_tasks if t not in maeb_tasks]
    if excluded_tasks:
        excluded_unique = get_unique_coverage_tasks(excluded_tasks)
        for dim, task_list in excluded_unique.items():
            if task_list:
                print(f"\n  {dim}:")
                for task, coverage in task_list:
                    print(f"    - {task} (unique: {coverage})")


def main():
    """Run the MAEB task selection analysis."""
    # Load benchmarks
    maeb_extended = mteb.get_benchmark("MAEB(extended)")

    # Start from MAEB(extended) as the source pool
    source_task_names = [t.metadata.name for t in maeb_extended.tasks]

    print(f"Source pool - MAEB(extended) tasks: {len(source_task_names)}")

    # Get metadata for all extended tasks
    metadata_df = get_task_metadata_df("MAEB(extended)")

    # Try to load model results for correlation analysis
    results_dir = Path("/Users/isaac/work/maeb-results/results")
    corr_matrix = None
    results_df = None

    if results_dir.exists():
        print("\nLoading model results for correlation analysis...")
        try:
            results_df, tasks_with_results = load_model_results(results_dir)

            # Filter to tasks in extended benchmark
            available_tasks = [t for t in source_task_names if t in tasks_with_results]
            print(
                f"Tasks with results: {len(available_tasks)}/{len(source_task_names)}"
            )

            if available_tasks:
                corr_matrix = compute_task_correlation(results_df, available_tasks)
                print(
                    f"Computed {len(corr_matrix)}x{len(corr_matrix)} correlation matrix"
                )
        except Exception as e:
            print(f"Warning: Could not load model results: {e}")
            print("Continuing without correlation analysis...")
    else:
        print(f"\nResults directory not found: {results_dir}")
        print("Skipping correlation analysis...")

    # Run iterative task selection if we have correlation data
    if results_df is not None and corr_matrix is not None:
        # Filter out excluded tasks
        filtered_source_tasks = [
            t for t in source_task_names if t not in TASKS_TO_EXCLUDE
        ]
        excluded_count = len(source_task_names) - len(filtered_source_tasks)

        # Build output for markdown file
        output_lines = []
        output_lines.append("# MAEB Task Selection Analysis")
        output_lines.append("")
        output_lines.append("## Overview")
        output_lines.append(
            f"- **Source pool**: MAEB(extended) with {len(source_task_names)} tasks"
        )
        if excluded_count > 0:
            output_lines.append(
                f"- **Excluded tasks**: {excluded_count} ({', '.join(TASKS_TO_EXCLUDE)})"
            )
        output_lines.append(f"- **Working pool**: {len(filtered_source_tasks)} tasks")
        output_lines.append(
            f"- **Goal**: Select non-redundant tasks while preserving coverage"
        )
        output_lines.append("")

        output_lines.append("## Selection Rules")
        output_lines.append("")
        output_lines.append(
            "1. **Retrieval direction preference**: For task families with both A2T and T2A, prefer T2A (text-to-audio)"
        )
        output_lines.append(
            "2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task"
        )
        output_lines.append(
            "3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage"
        )
        output_lines.append("")

        # Get protected tasks (unique coverage) from filtered source pool
        unique_tasks = get_unique_coverage_tasks(filtered_source_tasks)
        protected = set()
        for dim, task_list in unique_tasks.items():
            for task, coverage in task_list:
                protected.add(task)

        # Add manually protected tasks
        for task in MANUALLY_PROTECTED_TASKS:
            if task in filtered_source_tasks:
                protected.add(task)

        output_lines.append(f"## Protected Tasks (Unique Coverage): {len(protected)}")
        output_lines.append("")
        for dim, task_list in unique_tasks.items():
            if task_list:
                output_lines.append(f"### {dim.replace('_', ' ').title()}")
                for task, coverage in task_list[:20]:  # Limit to first 20
                    output_lines.append(f"- {task} (unique: {coverage})")
                if len(task_list) > 20:
                    output_lines.append(f"- ... and {len(task_list) - 20} more")
                output_lines.append("")

        # Show manually protected tasks
        manual_in_source = [
            t for t in MANUALLY_PROTECTED_TASKS if t in source_task_names
        ]
        if manual_in_source:
            output_lines.append("### Manually Protected")
            for task in manual_in_source:
                output_lines.append(f"- {task}")
            output_lines.append("")

        # Task families analysis
        output_lines.append("## Task Families (Same Source Dataset)")
        output_lines.append("")
        families = identify_task_families(metadata_df)
        for family, tasks in sorted(families.items(), key=lambda x: -len(x[1])):
            if len(tasks) > 1:
                output_lines.append(f"### {family} ({len(tasks)} tasks)")
                for task in tasks:
                    task_meta = metadata_df[metadata_df["name"] == task].iloc[0]
                    output_lines.append(
                        f"- {task} ({task_meta['type']}, {task_meta['category']})"
                    )
                output_lines.append("")

        # Run selection at different thresholds
        results_by_threshold = {}
        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, removed = iterative_task_selection(
                results_df,
                filtered_source_tasks,
                metadata_df,
                threshold=threshold,
                protected_tasks=protected,
                prefer_remove_same_source=True,
            )
            # Post-processing: enforce retrieval direction preference (T2A over A2T)
            remaining, direction_removed = enforce_retrieval_direction_preference(
                remaining
            )
            removed = removed + direction_removed

            # Post-processing: deduplicate same-source families
            remaining, family_removed = deduplicate_same_source_families(
                remaining, results_df, metadata_df
            )
            removed = removed + family_removed
            results_by_threshold[threshold] = (remaining, removed)

        # Compute correlations for each threshold (compare against full source, not filtered)
        correlations_by_threshold = {}
        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, _ = results_by_threshold[threshold]
            spearman, pearson = compute_benchmark_correlation(
                results_df, source_task_names, remaining
            )
            correlations_by_threshold[threshold] = (spearman, pearson)

        # Load evaluation times for all models
        all_model_eval_times = {}
        source_eval_times = {}
        for short_name, model_name in EVAL_TIME_MODELS:
            model_times = load_eval_times(
                results_dir, model_name, filtered_source_tasks
            )
            if model_times:
                all_model_eval_times[short_name] = model_times
                total, count = compute_total_eval_time(
                    model_times, filtered_source_tasks
                )
                source_eval_times[short_name] = (total, count)

        if all_model_eval_times:
            output_lines.append("## Evaluation Time (MAEB Extended Working Pool)")
            output_lines.append("")
            output_lines.append("| Model | Time | Tasks w/ data |")
            output_lines.append("|-------|------|---------------|")
            for short_name, model_name in EVAL_TIME_MODELS:
                if short_name in source_eval_times:
                    total, count = source_eval_times[short_name]
                    output_lines.append(
                        f"| {short_name} ({model_name}) | {format_duration(total)} | {count}/{len(filtered_source_tasks)} |"
                    )
                else:
                    output_lines.append(
                        f"| {short_name} ({model_name}) | N/A | 0/{len(filtered_source_tasks)} |"
                    )
            output_lines.append("")
        else:
            output_lines.append("## Evaluation Time")
            output_lines.append("")
            output_lines.append("*No evaluation time data found*")
            output_lines.append("")

        # Task type short names
        type_short = {
            "Any2AnyRetrieval": "Retr",
            "AudioClassification": "Class",
            "AudioClustering": "Clust",
            "AudioMultilabelClassification": "MLC",
            "AudioPairClassification": "Pair",
            "AudioReranking": "Rerank",
            "AudioZeroshotClassification": "ZS",
        }

        # Summary table
        output_lines.append("## Selection Results Summary")
        output_lines.append("")
        model_short_names = [short for short, _ in EVAL_TIME_MODELS]
        if all_model_eval_times:
            model_headers = " | ".join(model_short_names)
            output_lines.append(
                f"| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | Rerank | ZS | Langs | Doms | Spearman | Pearson | {model_headers} |"
            )
            model_sep = " | ".join(["---"] * len(model_short_names))
            output_lines.append(
                f"|-----------|-------|------|-------|-------|-----|------|--------|-------|-------|------|----------|---------|{model_sep}|"
            )
        else:
            output_lines.append(
                "| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | Rerank | ZS | Langs | Doms | Spearman | Pearson |"
            )
            output_lines.append(
                "|-----------|-------|------|-------|-------|-----|------|--------|-------|-------|------|----------|---------|"
            )

        original_coverage = get_coverage_analysis(filtered_source_tasks)

        # Compute eval times for each threshold (for all models)
        eval_times_by_threshold = {}
        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, _ = results_by_threshold[threshold]
            threshold_times = {}
            for short_name in model_short_names:
                if short_name in all_model_eval_times:
                    total_time, _ = compute_total_eval_time(
                        all_model_eval_times[short_name], remaining
                    )
                    threshold_times[short_name] = total_time
                else:
                    threshold_times[short_name] = None
            eval_times_by_threshold[threshold] = threshold_times

        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)
            spearman, pearson = correlations_by_threshold[threshold]
            threshold_times = eval_times_by_threshold[threshold]

            # Get type counts
            type_counts = coverage.get("type_counts", {})
            retr = type_counts.get("Any2AnyRetrieval", 0)
            cls = type_counts.get("AudioClassification", 0)
            clust = type_counts.get("AudioClustering", 0)
            mlc = type_counts.get("AudioMultilabelClassification", 0)
            pair = type_counts.get("AudioPairClassification", 0)
            rerank = type_counts.get("AudioReranking", 0)
            zs = type_counts.get("AudioZeroshotClassification", 0)

            base_row = (
                f"| {threshold} | {len(remaining)} | {retr} | {cls} | {clust} | {mlc} | {pair} | {rerank} | {zs} | "
                f"{coverage['n_languages']} | {coverage['n_domains']} | "
                f"{spearman:.4f} | {pearson:.4f}"
            )

            if all_model_eval_times:
                time_cols = []
                for short_name in model_short_names:
                    t = threshold_times.get(short_name)
                    if t is not None:
                        time_cols.append(format_duration(t))
                    else:
                        time_cols.append("N/A")
                output_lines.append(f"{base_row} | {' | '.join(time_cols)} |")
            else:
                output_lines.append(f"{base_row} |")

        output_lines.append("")
        output_lines.append(
            f"*Working pool: {len(filtered_source_tasks)} tasks, "
            f"{original_coverage['n_languages']} langs, "
            f"{original_coverage['n_domains']} doms*"
        )
        output_lines.append("")
        output_lines.append(
            "*Spearman/Pearson: Correlation of average model scores between selected tasks and full MAEB(extended)*"
        )
        output_lines.append("")

        # Detailed results for each threshold
        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)

            output_lines.append(f"## Threshold {threshold}")
            output_lines.append("")
            output_lines.append(
                f"**{len(source_task_names)} → {len(remaining)} tasks** ({len(removed)} removed)"
            )
            output_lines.append("")

            if removed:
                output_lines.append("### Removed Tasks")
                output_lines.append("")
                for task, reason, corr in removed:
                    output_lines.append(f"- {task}: {reason}")
                output_lines.append("")

            # VoxPopuli status
            vox_remaining = [t for t in remaining if "VoxPopuli" in t]
            output_lines.append(f"### VoxPopuli Tasks Remaining: {len(vox_remaining)}")
            for t in vox_remaining:
                output_lines.append(f"- {t}")
            output_lines.append("")

            # Coverage
            output_lines.append("### Coverage")
            output_lines.append(
                f"- Languages: {coverage['n_languages']} (was {original_coverage['n_languages']})"
            )
            output_lines.append(
                f"- Domains: {coverage['n_domains']} (was {original_coverage['n_domains']})"
            )
            output_lines.append(
                f"- Categories: {coverage['n_categories']} (was {original_coverage['n_categories']})"
            )
            output_lines.append(
                f"- Types: {coverage['n_types']} (was {original_coverage['n_types']})"
            )
            output_lines.append("")

        # Recommended task list (threshold 0.9)
        recommended_threshold = 0.8
        remaining, removed = results_by_threshold[recommended_threshold]

        output_lines.append(
            f"## Recommended MAEB Task List (threshold={recommended_threshold})"
        )
        output_lines.append("")
        output_lines.append(f"**Total: {len(remaining)} tasks**")
        output_lines.append("")

        # Group by type
        remaining_tasks = mteb.get_tasks(tasks=remaining)
        by_type = defaultdict(list)
        for task in remaining_tasks:
            by_type[task.metadata.type].append(task.metadata.name)

        for task_type, tasks in sorted(by_type.items()):
            output_lines.append(f"### {task_type} ({len(tasks)})")
            for t in sorted(tasks):
                task_obj = mteb.get_task(t)
                meta = task_obj.metadata
                langs = ", ".join(meta.languages[:3]) if meta.languages else "N/A"
                if meta.languages and len(meta.languages) > 3:
                    langs += f" (+{len(meta.languages) - 3} more)"
                domains = ", ".join(meta.domains) if meta.domains else "N/A"
                output_lines.append(f"- **{t}** - {meta.category}, {domains}")
            output_lines.append("")

        # Code block for benchmarks.py
        output_lines.append("### Code for benchmarks.py")
        output_lines.append("")
        output_lines.append("```python")
        output_lines.append("tasks=get_tasks(")
        output_lines.append("    tasks=[")
        for task_type, tasks in sorted(by_type.items()):
            output_lines.append(f"        # {task_type} ({len(tasks)})")
            for t in sorted(tasks):
                output_lines.append(f'        "{t}",')
        output_lines.append("    ]")
        output_lines.append("),")
        output_lines.append("```")
        output_lines.append("")

        # Write to markdown file
        output_path = Path("scripts/maeb_task_selection_analysis.md")
        output_path.write_text("\n".join(output_lines))
        print(f"\nAnalysis written to: {output_path}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nSource: MAEB(extended) with {len(source_task_names)} tasks")
        if excluded_count > 0:
            print(f"Excluded: {excluded_count} tasks ({', '.join(TASKS_TO_EXCLUDE)})")
        print(f"Working pool: {len(filtered_source_tasks)} tasks")
        if source_eval_times:
            print("\nEval times for working pool:")
            for short_name, model_name in EVAL_TIME_MODELS:
                if short_name in source_eval_times:
                    total, count = source_eval_times[short_name]
                    print(
                        f"  {short_name}: {format_duration(total)} ({count}/{len(filtered_source_tasks)} tasks)"
                    )
        print(f"\nProtected tasks: {len(protected)}")
        print("\nResults by threshold:")
        # Simplified console output without per-model times
        print(
            f"  {'Thresh':<7} {'Tasks':<6} {'Retr':<5} {'Cls':<4} {'Clu':<4} {'MLC':<4} {'Pair':<5} {'Rer':<4} {'ZS':<3} {'Spearman':<9} {'Pearson':<8}"
        )
        print(
            f"  {'-' * 7} {'-' * 6} {'-' * 5} {'-' * 4} {'-' * 4} {'-' * 4} {'-' * 5} {'-' * 4} {'-' * 3} {'-' * 9} {'-' * 8}"
        )
        for threshold in [0.95, 0.93, 0.9, 0.8, 0.7, 0.6]:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)
            type_counts = coverage.get("type_counts", {})
            spearman, pearson = correlations_by_threshold[threshold]
            retr = type_counts.get("Any2AnyRetrieval", 0)
            cls = type_counts.get("AudioClassification", 0)
            clust = type_counts.get("AudioClustering", 0)
            mlc = type_counts.get("AudioMultilabelClassification", 0)
            pair = type_counts.get("AudioPairClassification", 0)
            rerank = type_counts.get("AudioReranking", 0)
            zs = type_counts.get("AudioZeroshotClassification", 0)
            print(
                f"  {threshold:<7} {len(remaining):<6} {retr:<5} {cls:<4} {clust:<4} {mlc:<4} {pair:<5} {rerank:<4} {zs:<3} {spearman:<9.4f} {pearson:<8.4f}"
            )

        print(
            f"\nRecommended (threshold={recommended_threshold}): {len(results_by_threshold[recommended_threshold][0])} tasks"
        )
        rec_spearman, rec_pearson = correlations_by_threshold[recommended_threshold]
        print(
            f"  Correlation with MAEB(extended): Spearman={rec_spearman:.4f}, Pearson={rec_pearson:.4f}"
        )
        print(f"\nFull analysis (with per-model eval times) saved to: {output_path}")


if __name__ == "__main__":
    main()
