import re
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

import mteb
from mteb.get_tasks import get_task, get_tasks
from mteb.results.benchmark_results import BenchmarkResults


def _borda_count(scores: pd.Series) -> pd.Series:
    n = len(scores)
    ranks = scores.rank(method="average", ascending=False)
    counts = n - ranks
    return counts


def _get_borda_rank(score_table: pd.DataFrame) -> pd.Series:
    borda_counts = score_table.apply(_borda_count, axis="index")
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


def _format_max_tokens(max_tokens: float | None) -> float | None:
    if max_tokens is None or max_tokens == np.inf:
        return None
    return float(max_tokens)


def _get_means_per_types(per_task: pd.DataFrame):
    task_names_per_type = defaultdict(list)
    for task_name in per_task.columns:
        task_type = get_task(task_name).metadata.type
        task_names_per_type[task_type].append(task_name)
    records = []
    for task_type, tasks in task_names_per_type.items():
        for model_name, scores in per_task.iterrows():
            records.append(
                dict(
                    model_name=model_name,
                    task_type=task_type,
                    score=scores[tasks].mean(skipna=False),
                )
            )
    return pd.DataFrame.from_records(records)


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
    joint_table = joint_table.drop(models_to_remove, axis=0)
    joint_table.insert(0, "mean", overall_mean)
    joint_table.insert(1, "mean_by_task_type", typed_mean)
    joint_table["borda_rank"] = _get_borda_rank(per_task)
    joint_table = joint_table.sort_values("borda_rank", ascending=True)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(mteb.get_model_meta)
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
        model_metas.map(lambda m: int(m.embed_dim) if m.embed_dim else None),
    )
    joint_table.insert(
        1,
        "Number of Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Memory Usage (MB)",
        model_metas.map(
            lambda m: int(m.memory_usage_mb) if m.memory_usage_mb else None
        ),
    )

    # Add zero-shot percentage
    tasks = get_tasks(tasks=list(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: m.zero_shot_percentage(tasks))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

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
    joint_table = joint_table.drop(models_to_remove, axis=0)
    joint_table.insert(0, "mean(public)", public_mean)
    joint_table.insert(1, "mean(private)", private_mean)
    joint_table["borda_rank"] = _get_borda_rank(per_task)
    joint_table = joint_table.sort_values("borda_rank", ascending=True)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(mteb.get_model_meta)
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
        model_metas.map(lambda m: int(m.embed_dim) if m.embed_dim else None),
    )
    joint_table.insert(
        1,
        "Number of Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Memory Usage (MB)",
        model_metas.map(
            lambda m: int(m.memory_usage_mb) if m.memory_usage_mb else None
        ),
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
    joint_table = joint_table.drop(models_to_remove, axis=0)
    joint_table.insert(0, "mean(subset)", overall_subset_mean)
    joint_table["borda_rank"] = _get_borda_rank(per_subset)
    joint_table = joint_table.sort_values("mean(subset)", ascending=False)
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(mteb.get_model_meta)
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
        model_metas.map(lambda m: int(m.embed_dim) if m.embed_dim else None),
    )
    joint_table.insert(
        1,
        "Number of Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Memory Usage (MB)",
        model_metas.map(
            lambda m: int(m.memory_usage_mb) if m.memory_usage_mb else None
        ),
    )

    # Add zero-shot percentage
    tasks = get_tasks(tasks=list(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: m.zero_shot_percentage(tasks))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

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

    # Build joint table
    joint_table = mean_per_type.copy()
    joint_table = joint_table.drop(models_to_remove, axis=0)
    joint_table.insert(0, "mean_by_task_type", typed_mean)
    joint_table = joint_table.sort_values("mean_by_task_type", ascending=False)
    joint_table["borda_rank"] = _get_borda_rank(per_task)
    joint_table["rank"] = [i + 1 for i in range(len(joint_table))]
    joint_table = joint_table.reset_index()

    # Add model metadata
    model_metas = joint_table["model_name"].map(mteb.get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)

    # Insert model metadata columns
    joint_table.insert(
        1, "Max Tokens", model_metas.map(lambda m: _format_max_tokens(m.max_tokens))
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: int(m.embed_dim) if m.embed_dim else None),
    )
    joint_table.insert(
        1,
        "Number of Parameters (B)",
        model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
    )
    joint_table.insert(
        1,
        "Memory Usage (MB)",
        model_metas.map(
            lambda m: int(m.memory_usage_mb) if m.memory_usage_mb else None
        ),
    )

    # Add zero-shot percentage
    tasks = get_tasks(tasks=list(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: m.zero_shot_percentage(tasks))
    )
    joint_table["Zero-shot"] = joint_table["Zero-shot"].fillna(-1)

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
