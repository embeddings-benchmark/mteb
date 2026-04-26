from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from mteb.get_tasks import _TASKS_REGISTRY, get_tasks
from mteb.models.get_model_meta import get_model_meta

if TYPE_CHECKING:
    from mteb.results.benchmark_results import BenchmarkResults


@functools.lru_cache(maxsize=128)
def _get_tasks_cached(task_names: tuple[str, ...]):
    """Memoized `get_tasks(tasks=...)` for table-creation hot paths."""
    return get_tasks(tasks=list(task_names))


@functools.lru_cache(maxsize=4096)
def _zero_shot_pct_cached(model_name: str, task_names: tuple[str, ...]) -> int | None:
    """Memoized zero_shot_percentage — expensive due to task-similarity graph traversal."""
    meta = get_model_meta(model_name)
    if meta is None:
        return None
    return meta.zero_shot_percentage(_get_tasks_cached(task_names))


def _get_borda_rank(score_table: pd.DataFrame) -> pd.Series:
    n = len(score_table)
    borda_counts = n - score_table.rank(method="average", ascending=False, axis=0)
    mean_borda = borda_counts.sum(axis=1)
    return mean_borda.rank(method="min", ascending=False).astype(int)


def _split_on_capital(s: str) -> str:
    """Splits on capital letters and joins with spaces

    Returns:
        The input string split on capital letters and joined with spaces as a string.
    """
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s))


def _format_n_parameters(n_parameters) -> float | None:
    """Format n_parameters to be in billions with decimals down to 1 million. I.e. 7M -> 0.007B, 1.5B -> 1.5B, None -> None"""
    if n_parameters:
        n_parameters = float(n_parameters)
        return round(n_parameters / 1e9, 3)
    return None


def _format_n_active_parameters(n_active_parameters) -> float | None:
    """Format n_active_parameters to be in billions with decimals down to 1 million. I.e. 7M -> 0.007B, 1.5B -> 1.5B, None -> None"""
    if n_active_parameters is not None:
        n_active_parameters = float(n_active_parameters)
        return round(n_active_parameters / 1e9, 3)
    return None


def _format_max_tokens(max_tokens: float | None) -> float | None:
    if max_tokens is None or max_tokens == np.inf:
        return None
    return float(max_tokens)


def _get_embedding_size(embed_dim: int | list[int] | None) -> int | None:
    if embed_dim is None:
        return None
    if isinstance(embed_dim, int):
        return int(embed_dim)
    if isinstance(embed_dim, Sequence) and len(embed_dim) > 0:
        return int(max(embed_dim))
    return None


def _get_means_per_types(per_task: pd.DataFrame):
    task_names_per_type = defaultdict(list)
    for task_name in per_task.columns:
        # Read from the registered class to skip instantiation (get_task() runs filter_languages()).
        task_type = _TASKS_REGISTRY[task_name].metadata.type
        task_names_per_type[task_type].append(task_name)

    type_means = {
        task_type: per_task[tasks].mean(axis=1, skipna=False)
        for task_type, tasks in task_names_per_type.items()
    }
    wide = pd.DataFrame(type_means)
    wide.index.name = "model_name"
    return wide.reset_index().melt(
        id_vars="model_name", var_name="task_type", value_name="score"
    )


def _create_summary_table_from_benchmark_results(
    benchmark_results: BenchmarkResults,
) -> pd.DataFrame:
    """Create summary table from BenchmarkResults.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages.

    Args:
        benchmark_results: BenchmarkResults object containing model results

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard
    """
    data = benchmark_results.to_dataframe(format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    # Convert to DataFrame and pivot
    per_task = data.pivot(index="model_name", columns="task_name", values="score")

    # Remove models with no scores
    to_remove = per_task.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_task[to_remove].index)
    per_task = per_task.drop(models_to_remove, axis=0)

    # Calculate means by task type
    mean_per_type = _get_means_per_types(per_task)
    mean_per_type = mean_per_type.pivot(
        index="model_name", columns="task_type", values="score"
    )
    mean_per_type.columns = [
        _split_on_capital(column) for column in mean_per_type.columns
    ]

    # Calculate overall means
    typed_mean = mean_per_type.mean(skipna=False, axis=1)
    overall_mean = per_task.mean(skipna=False, axis=1)

    # Build joint table
    joint_table = mean_per_type.copy()
    joint_table.insert(0, "mean", overall_mean)
    joint_table.insert(1, "mean_by_task_type", typed_mean)
    joint_table["borda_rank"] = _get_borda_rank(per_task)
    joint_table = joint_table.sort_values("borda_rank", ascending=True)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)

    # Insert model metadata columns
    joint_table.insert(
        1,
        "Max Tokens",
        model_metas.map(lambda m: _format_max_tokens(m.max_tokens)),
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: _get_embedding_size(m.embed_dim)),
    )
    joint_table.insert(
        1,
        "Total Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Active Parameters (B)",
        model_metas.map(lambda m: _format_n_active_parameters(m.n_active_parameters)),
    )

    # Add zero-shot percentage
    _task_names_key = tuple(sorted(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: _zero_shot_pct_cached(m.name, _task_names_key))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

    # Add release date from model metadata
    joint_table["Release Date"] = model_metas.map(
        lambda m: str(m.release_date) if m.release_date else None
    )

    # Clean up model names (remove HF organization)
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )

    # Add markdown links to model names
    name_w_link = (
        "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
    )
    joint_table["model_name"] = joint_table["model_name"].mask(
        joint_table["model_link"].notna(), name_w_link
    )
    joint_table = joint_table.drop(columns=["model_link"])

    # Rename columns
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": "Mean (TaskType)",
            "mean": "Mean (Task)",
        }
    )

    # Move borda rank to front
    joint_table.insert(0, "Rank (Borda)", joint_table.pop("borda_rank"))

    return joint_table


def _create_per_task_table_from_benchmark_results(
    benchmark_results: BenchmarkResults,
) -> pd.DataFrame:
    """Create per-task table from BenchmarkResults.

    Returns a DataFrame with one row per model and one column per task.

    Args:
        benchmark_results: BenchmarkResults object containing model results

    Returns:
        DataFrame with per-task scores, ready for styling in the leaderboard
    """
    # Get scores in long format
    data = benchmark_results.to_dataframe(format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    # Convert to DataFrame and pivot
    per_task = data.pivot(index="model_name", columns="task_name", values="score")

    # Remove models with no scores
    to_remove = per_task.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_task[to_remove].index)
    per_task = per_task.drop(models_to_remove, axis=0)

    # Add borda rank and sort
    per_task["borda_rank"] = _get_borda_rank(per_task)
    per_task = per_task.sort_values("borda_rank", ascending=True)
    per_task = per_task.drop(columns=["borda_rank"])
    per_task = per_task.reset_index()

    # Clean up model names (remove HF organization)
    per_task["model_name"] = per_task["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    per_task = per_task.rename(
        columns={
            "model_name": "Model",
        }
    )

    return per_task


def _create_per_language_table_from_benchmark_results(
    benchmark_results: BenchmarkResults,
    language_view: list[str] | Literal["all"],
) -> pd.DataFrame:
    """Create per-language table from BenchmarkResults.

    Returns a DataFrame with one row per model and one column per language.

    Args:
        benchmark_results: BenchmarkResults object containing model results
        language_view: List of languages to include in the per-language table, or "all" for all languages present in the results
    Returns:
        DataFrame with per-language scores, ready for styling in the leaderboard
    """
    if language_view != "all" and not isinstance(language_view, list):
        raise ValueError("language_view must be a list of languages or 'all'")

    data = benchmark_results.to_dataframe(aggregation_level="language", format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    if language_view != "all":
        data = data[data["language"].isin(language_view)]

    per_language = data.pivot_table(
        index="model_name", columns="language", values="score", aggfunc="mean"
    )

    to_remove = per_language.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_language[to_remove].index)
    per_language = per_language.drop(models_to_remove, axis=0)

    per_language["borda_rank"] = _get_borda_rank(per_language)
    per_language = per_language.sort_values("borda_rank", ascending=True)
    per_language = per_language.drop(columns=["borda_rank"])
    per_language = per_language.reset_index()

    per_language["model_name"] = per_language["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    per_language = per_language.rename(
        columns={
            "model_name": "Model",
        }
    )

    return per_language


def _create_summary_table_mean_public_private(
    benchmark_results: BenchmarkResults,
    exclude_private_from_borda: bool = False,
) -> pd.DataFrame:
    """Create summary table from BenchmarkResults.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages.

    Args:
        benchmark_results: BenchmarkResults object containing model results
        exclude_private_from_borda: If True, calculate Borda rank using only public tasks

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard
    """
    data = benchmark_results.to_dataframe(format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame
    public_task_name = benchmark_results._filter_tasks(is_public=True).task_names
    private_task_name = benchmark_results._filter_tasks(is_public=False).task_names
    # Convert to DataFrame and pivot
    per_task = data.pivot(index="model_name", columns="task_name", values="score")

    # Remove models with no scores
    to_remove = per_task.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_task[to_remove].index)
    per_task = per_task.drop(models_to_remove, axis=0)

    # Calculate means by task type
    mean_per_type = _get_means_per_types(per_task)
    mean_per_type = mean_per_type.pivot(
        index="model_name", columns="task_type", values="score"
    )
    mean_per_type.columns = [
        _split_on_capital(column) for column in mean_per_type.columns
    ]

    # Calculate overall means
    public_mean = per_task[public_task_name].mean(skipna=False, axis=1)
    private_mean = per_task[private_task_name].mean(skipna=False, axis=1)

    # Build joint table
    joint_table = mean_per_type.copy()
    joint_table.insert(0, "mean(public)", public_mean)
    joint_table.insert(1, "mean(private)", private_mean)
    if exclude_private_from_borda:
        borda_per_task = per_task[public_task_name]
    else:
        borda_per_task = per_task
    joint_table["borda_rank"] = _get_borda_rank(borda_per_task)
    joint_table = joint_table.sort_values("borda_rank", ascending=True)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)

    # Insert model metadata columns
    joint_table.insert(
        1,
        "Max Tokens",
        model_metas.map(lambda m: _format_max_tokens(m.max_tokens)),
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: _get_embedding_size(m.embed_dim)),
    )
    joint_table.insert(
        1,
        "Total Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Active Parameters (B)",
        model_metas.map(lambda m: _format_n_active_parameters(m.n_active_parameters)),
    )

    # Add release date from model metadata
    joint_table["Release Date"] = model_metas.map(
        lambda m: str(m.release_date) if m.release_date else None
    )

    # Clean up model names (remove HF organization)
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )

    # Add markdown links to model names
    name_w_link = (
        "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
    )
    joint_table["model_name"] = joint_table["model_name"].mask(
        joint_table["model_link"].notna(), name_w_link
    )
    joint_table = joint_table.drop(columns=["model_link"])

    # Rename columns
    rename_dict = {
        "model_name": "Model",
        "mean(public)": "Mean (Public)",
        "mean(private)": "Mean (Private)",
    }

    joint_table = joint_table.rename(columns=rename_dict)

    # Move borda rank to front
    joint_table.insert(0, "Rank (Borda)", joint_table.pop("borda_rank"))

    return joint_table


def _create_summary_table_mean_subset(
    benchmark_results: BenchmarkResults,
) -> pd.DataFrame:
    """Create summary table from BenchmarkResults.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages. Calculates means where each task-language subset
    is weighted equally.

    Args:
        benchmark_results: BenchmarkResults object containing model results

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard
    """
    data = benchmark_results.to_dataframe(format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    # Convert to DataFrame and pivot
    per_task = data.pivot(index="model_name", columns="task_name", values="score")

    # Remove models with no scores
    to_remove = per_task.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_task[to_remove].index)
    per_task = per_task.drop(models_to_remove, axis=0)

    # Calculate means by task type
    mean_per_type = _get_means_per_types(per_task)
    mean_per_type = mean_per_type.pivot(
        index="model_name", columns="task_type", values="score"
    )
    mean_per_type.columns = [
        _split_on_capital(column) for column in mean_per_type.columns
    ]

    # Calculate subset means (each task-language combination weighted equally)
    detailed_data = benchmark_results.to_dataframe(
        aggregation_level="subset", format="long"
    )
    overall_subset_mean = detailed_data.groupby("model_name")["score"].mean()

    per_subset = detailed_data.pivot(
        index="model_name", columns=["task_name", "subset"], values="score"
    )

    # Build joint table
    joint_table = mean_per_type.copy()
    joint_table.insert(0, "mean(subset)", overall_subset_mean)
    joint_table["borda_rank"] = _get_borda_rank(per_subset)
    joint_table = joint_table.sort_values("mean(subset)", ascending=False)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)

    # Insert model metadata columns
    joint_table.insert(
        1,
        "Max Tokens",
        model_metas.map(lambda m: _format_max_tokens(m.max_tokens)),
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: _get_embedding_size(m.embed_dim)),
    )
    joint_table.insert(
        1,
        "Total Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Active Parameters (B)",
        model_metas.map(lambda m: _format_n_active_parameters(m.n_active_parameters)),
    )
    # Add zero-shot percentage
    _task_names_key = tuple(sorted(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: _zero_shot_pct_cached(m.name, _task_names_key))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

    # Add release date from model metadata
    joint_table["Release Date"] = model_metas.map(
        lambda m: str(m.release_date) if m.release_date else None
    )

    # Clean up model names (remove HF organization)
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )

    # Add markdown links to model names
    name_w_link = (
        "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
    )
    joint_table["model_name"] = joint_table["model_name"].mask(
        joint_table["model_link"].notna(), name_w_link
    )
    joint_table = joint_table.drop(columns=["model_link"])

    # Rename columns
    rename_dict = {
        "model_name": "Model",
        "mean(subset)": "Mean (Subset)",
    }
    joint_table = joint_table.rename(columns=rename_dict)

    # Move borda rank to front
    joint_table.insert(0, "Rank (Borda)", joint_table.pop("borda_rank"))

    return joint_table


def _create_summary_table_mean_task_type(
    benchmark_results: BenchmarkResults, mean_column_name: str = "Mean (TaskType)"
) -> pd.DataFrame:
    """Create summary table from BenchmarkResults.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages.

    Args:
        benchmark_results: BenchmarkResults object containing model results
        mean_column_name: Name for the mean-by-task-type column. Defaults to "Mean (TaskType)".

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard
    """
    data = benchmark_results.to_dataframe(format="long")

    if data.empty:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    # Convert to DataFrame and pivot
    per_task = data.pivot(index="model_name", columns="task_name", values="score")

    # Remove models with no scores
    to_remove = per_task.isna().all(axis="columns")
    if to_remove.all():
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return no_results_frame

    models_to_remove = list(per_task[to_remove].index)
    per_task = per_task.drop(models_to_remove, axis=0)

    # Calculate means by task type
    mean_per_type = _get_means_per_types(per_task)
    mean_per_type = mean_per_type.pivot(
        index="model_name", columns="task_type", values="score"
    )
    mean_per_type.columns = [
        _split_on_capital(column) for column in mean_per_type.columns
    ]

    # Calculate overall means
    typed_mean = mean_per_type.mean(skipna=False, axis=1)

    # Build joint table
    joint_table = mean_per_type.copy()
    joint_table.insert(0, "mean_by_task_type", typed_mean)
    joint_table = joint_table.sort_values("mean_by_task_type", ascending=False)
    joint_table["borda_rank"] = _get_borda_rank(per_task)
    joint_table["rank"] = [i + 1 for i in range(len(joint_table))]
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)

    # Insert model metadata columns
    joint_table.insert(
        1, "Max Tokens", model_metas.map(lambda m: _format_max_tokens(m.max_tokens))
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: _get_embedding_size(m.embed_dim)),
    )
    joint_table.insert(
        1,
        "Total Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Active Parameters (B)",
        model_metas.map(lambda m: _format_n_active_parameters(m.n_active_parameters)),
    )

    # Add zero-shot percentage
    _task_names_key = tuple(sorted(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: _zero_shot_pct_cached(m.name, _task_names_key))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

    # Add release date from model metadata
    joint_table["Release Date"] = model_metas.map(
        lambda m: str(m.release_date) if m.release_date else None
    )

    # Clean up model names (remove HF organization)
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )

    # Add markdown links to model names
    name_w_link = (
        "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
    )
    joint_table["model_name"] = joint_table["model_name"].mask(
        joint_table["model_link"].notna(), name_w_link
    )
    joint_table = joint_table.drop(columns=["model_link"])

    # Rename columns
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": mean_column_name,
            "borda_rank": "Rank (Borda)",
        }
    )

    if "Any Any Multilingual Retrieval" in joint_table.columns:
        joint_table = joint_table.rename(
            columns={"Any Any Multilingual Retrieval": "Multilingual Retrieval"}
        )
    if "Any Any Retrieval" in joint_table.columns:
        joint_table = joint_table.rename(columns={"Any Any Retrieval": "Retrieval"})

    # Move borda rank to front
    joint_table.insert(0, "Rank", joint_table.pop("rank"))

    return joint_table
