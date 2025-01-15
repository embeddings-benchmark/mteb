from __future__ import annotations

import math
import re
from collections import defaultdict

import gradio as gr
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mteb.models.overview import get_model_meta
from mteb.overview import get_task, get_tasks


def borda_count(scores: pd.Series) -> pd.Series:
    n = len(scores)
    ranks = scores.rank(method="average", ascending=False)
    counts = n - ranks
    return counts


def get_borda_rank(score_table: pd.DataFrame) -> pd.Series:
    borda_counts = score_table.apply(borda_count, axis="index")
    mean_borda = borda_counts.sum(axis=1)
    return mean_borda.rank(method="min", ascending=False).astype(int)


def format_scores(score: float) -> float:
    return round(score * 100, 2)


def format_n_parameters(n_parameters) -> str:
    if (n_parameters is None) or (not int(n_parameters)):
        return "Unknown"
    n_thousand = int(n_parameters // 1e3)
    if n_thousand < 1:
        return str(int(n_parameters))
    n_zeros = math.log10(n_thousand)
    if n_zeros >= 6:
        return str(n_thousand // (10**6)) + "B"
    if n_zeros >= 3:
        return str(n_thousand // (10**3)) + "M"
    return str(n_thousand) + "K"


def split_on_capital(s: str) -> str:
    """Splits on capital letters and joins with spaces"""
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s))


def get_column_widths(df: pd.DataFrame) -> list[str]:
    widths = []
    for column_name in df.columns:
        column_word_lengths = [len(word) for word in column_name.split()]
        if is_numeric_dtype(df[column_name]):
            value_lengths = [len(f"{value:.2f}") for value in df[column_name]]
        else:
            value_lengths = [len(str(value)) for value in df[column_name]]
        try:
            max_length = max(max(column_word_lengths), max(value_lengths))
            n_pixels = 35 + (max_length * 12.5)
            widths.append(f"{n_pixels}px")
        except Exception:
            widths.append("50px")
    return widths


def get_column_types(df: pd.DataFrame) -> list[str]:
    types = []
    for column_name in df.columns:
        if is_numeric_dtype(df[column_name]):
            types.append("number")
        else:
            types.append("str")
    return types


def get_means_per_types(per_task: pd.DataFrame):
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


def failsafe_get_model_meta(model_name):
    try:
        return get_model_meta(model_name)
    except Exception:
        return None


def format_max_tokens(max_tokens: float | None) -> str:
    if max_tokens is None:
        return "Unknown"
    if max_tokens == np.inf:
        return "Infinite"
    return str(int(max_tokens))


def get_zero_shot_emoji(model_meta, tasks):
    if model_meta is None:
        return "⚠️"
    is_zero_shot = model_meta.is_zero_shot_on(tasks)
    if is_zero_shot is None:
        return "⚠️"
    if is_zero_shot:
        return "✅"
    return "❌"


def scores_to_tables(
    scores_long: list[dict], search_query: str | None = None
) -> tuple[gr.DataFrame, gr.DataFrame]:
    if not scores_long:
        no_results_frame = pd.DataFrame(
            {"No results": ["You can try relaxing your criteria"]}
        )
        return gr.DataFrame(no_results_frame), gr.DataFrame(no_results_frame)
    data = pd.DataFrame.from_records(scores_long)
    per_task = data.pivot(index="model_name", columns="task_name", values="score")
    mean_per_type = get_means_per_types(per_task)
    mean_per_type = mean_per_type.pivot(
        index="model_name", columns="task_type", values="score"
    )
    mean_per_type.columns = [
        split_on_capital(column) for column in mean_per_type.columns
    ]
    to_remove = per_task.isna().all(axis="columns")
    if search_query:
        names = per_task.index.get_level_values("model_name")
        names = pd.Series(names, index=per_task.index)
        to_remove |= ~names.str.contains(search_query, regex=True)
    models_to_remove = list(per_task[to_remove].index)
    typed_mean = mean_per_type.mean(skipna=False, axis=1)
    overall_mean = per_task.mean(skipna=False, axis=1)
    joint_table = mean_per_type.copy()
    per_task = per_task.drop(models_to_remove, axis=0)
    joint_table = joint_table.drop(models_to_remove, axis=0)
    joint_table.insert(0, "mean", overall_mean)
    joint_table.insert(1, "mean_by_task_type", typed_mean)
    joint_table["borda_rank"] = get_borda_rank(per_task)
    joint_table = joint_table.sort_values("borda_rank", ascending=True)
    per_task["borda_rank"] = joint_table["borda_rank"]
    per_task = per_task.sort_values("borda_rank", ascending=True)
    per_task = per_task.drop(columns=["borda_rank"])
    joint_table = joint_table.reset_index()
    model_metas = joint_table["model_name"].map(failsafe_get_model_meta)
    joint_table = joint_table[model_metas.notna()]
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)
    joint_table.insert(
        1,
        "Max Tokens",
        model_metas.map(lambda m: format_max_tokens(m.max_tokens)),
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: str(int(m.embed_dim)) if m.embed_dim else "Unknown"),
    )
    joint_table.insert(
        1,
        "Number of Parameters",
        model_metas.map(lambda m: format_n_parameters(m.n_parameters)),
    )
    tasks = get_tasks(tasks=list(data["task_name"].unique()))
    joint_table.insert(
        1, "Zero-shot", model_metas.map(lambda m: get_zero_shot_emoji(m, tasks))
    )
    # Removing HF organization from model
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    # Adding markdown link to model names
    name_w_link = (
        "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
    )
    joint_table["model_name"] = joint_table["model_name"].mask(
        joint_table["model_link"].notna(), name_w_link
    )
    joint_table = joint_table.drop(columns=["model_link"])
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": "Mean (TaskType)",
            "mean": "Mean (Task)",
        }
    )
    per_task = per_task.reset_index()
    per_task["model_name"] = per_task["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    per_task = per_task.rename(
        columns={
            "model_name": "Model",
        }
    )
    joint_table.insert(0, "Rank (Borda)", joint_table.pop("borda_rank"))
    column_widths = get_column_widths(joint_table)
    task_column_widths = get_column_widths(per_task)
    # overriding for model name
    column_widths[1] = "250px"
    column_types = get_column_types(joint_table)
    # setting model name column to markdown
    column_types[1] = "markdown"
    score_columns = ["Mean (Task)", "Mean (TaskType)", *mean_per_type.columns]
    joint_table[score_columns] = joint_table[score_columns].map(format_scores)
    joint_table_style = (
        joint_table.style.format(
            {**{column: "{:.2f}" for column in score_columns}, "Rank (Borda)": "{:.0f}"}
        )
        .highlight_min("Rank (Borda)", props="font-weight: bold")
        .highlight_max(subset=score_columns, props="font-weight: bold")
    )
    task_score_columns = per_task.select_dtypes("number").columns
    per_task[task_score_columns] *= 100
    per_task_style = per_task.style.format(
        "{:.2f}", subset=task_score_columns
    ).highlight_max(subset=task_score_columns, props="font-weight: bold")
    return (
        gr.DataFrame(
            joint_table_style,
            column_widths=column_widths,
            datatype=column_types,
            interactive=False,
            wrap=True,
        ),
        gr.DataFrame(
            per_task_style, column_widths=task_column_widths, interactive=False
        ),
    )
