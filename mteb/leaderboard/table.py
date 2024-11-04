from __future__ import annotations

import math
import re

import gradio as gr
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from mteb.models.overview import get_model_meta
from mteb.overview import get_task


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
    return score * 100


def format_n_parameters(n_parameters) -> str:
    if n_parameters is None:
        return ""
    n_million = int(n_parameters) // 1e6
    n_zeros = math.log10(n_million)
    if n_zeros >= 3:
        return str(n_million // (10**3)) + "B"
    return str(n_million) + "M"


def split_on_capital(s: str) -> str:
    """Splits on capital letters and joins with spaces"""
    if all([c.isupper() for c in s]):
        return s
    return " ".join(re.findall("[A-Z][^A-Z]*", s))


def get_column_widths(df: pd.DataFrame) -> list[str]:
    widths = []
    for column_name in df.columns:
        column_word_lengths = [len(word) for word in column_name.split()]
        if is_numeric_dtype(df[column_name]):
            value_lengths = [len(f"{value:.2f}") for value in df[column_name]]
        else:
            value_lengths = [len(str(value)) for value in df[column_name]]
        max_length = max(max(column_word_lengths), max(value_lengths))
        n_pixels = 25 + (max_length * 10)
        widths.append(f"{n_pixels}px")
    return widths


def get_column_types(df: pd.DataFrame) -> list[str]:
    types = []
    for column_name in df.columns:
        if is_numeric_dtype(df[column_name]):
            types.append("number")
        else:
            types.append("str")
    return types


def scores_to_tables(
    scores_long: list[dict], search_query: str | None = None
) -> tuple[gr.DataFrame, gr.DataFrame]:
    if not scores_long:
        return gr.DataFrame(), gr.DataFrame()
    data = pd.DataFrame.from_records(scores_long)
    data["task_type"] = data["task_name"].map(
        lambda task_name: get_task(task_name).metadata.type
    )
    mean_per_type = (
        data.groupby(["model_name", "model_revision", "task_type"])[["score"]]
        .agg(np.nanmean)
        .reset_index()
    )
    typed_mean = (
        mean_per_type.groupby(["model_name", "model_revision"])[["score"]]
        .agg(np.nanmean)
        .rename(columns={"score": "mean_by_task_type"})
    )
    mean_per_type = mean_per_type.pivot(
        index=["model_name", "model_revision"], columns="task_type", values="score"
    )
    mean_per_type.columns = [
        split_on_capital(column) for column in mean_per_type.columns
    ]
    per_task = data.pivot(
        index=["model_name", "model_revision"], columns="task_name", values="score"
    )
    to_remove = per_task.isna().any(axis="columns")
    if search_query:
        names = per_task.index.get_level_values("model_name")
        names = pd.Series(names, index=per_task.index)
        to_remove |= ~names.str.contains(search_query, regex=True)
    overall_mean = (
        data.groupby(["model_name", "model_revision"])[["score"]]
        .agg(np.nanmean)
        .rename(columns={"score": "mean"})
    )
    per_task = per_task[~to_remove]
    mean_per_type = mean_per_type[~to_remove]
    overall_mean = overall_mean[~to_remove]
    joint_table = overall_mean.join([typed_mean, mean_per_type])
    joint_table["borda_rank"] = get_borda_rank(per_task)
    joint_table = joint_table.reset_index()
    joint_table = joint_table.drop(columns=["model_revision"])
    model_metas = joint_table["model_name"].map(get_model_meta)
    joint_table["model_link"] = model_metas.map(lambda m: m.reference)
    # joint_table.insert(
    #     1,
    #     "Rank (Mean)",
    #     joint_table["mean"].rank(ascending=False, method="min").astype(int),
    # )
    joint_table.insert(
        1,
        "Max Tokens",
        model_metas.map(lambda m: str(int(m.max_tokens)) if m.max_tokens else ""),
    )
    joint_table.insert(
        1,
        "Embedding Dimensions",
        model_metas.map(lambda m: str(int(m.embed_dim)) if m.embed_dim else ""),
    )
    joint_table.insert(
        1,
        "Number of Parameters",
        model_metas.map(lambda m: format_n_parameters(m.n_parameters)),
    )
    joint_table = joint_table.sort_values("mean", ascending=False)
    # Removing HF organization from model
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    # Adding markdown link to model names
    joint_table["model_name"] = (
        "[" + joint_table["model_name"] + "](" + joint_table.pop("model_link") + ")"
    )
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": "Mean (TaskType)",
            "mean": "Mean (Task)",
        }
    )
    per_task = per_task.reset_index().drop(columns=["model_revision"])
    per_task["model_name"] = per_task["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    per_task = per_task.rename(
        columns={
            "model_name": "Model",
        }
    )
    joint_table.insert(0, "Rank (Borda)", joint_table.pop("borda_rank"))
    to_format = ["Mean (Task)", "Mean (TaskType)", *mean_per_type.columns]
    joint_table[to_format] = joint_table[to_format].map(format_scores)
    joint_table = joint_table.style.highlight_max(
        subset=to_format,
        props="font-weight: bold",
    )
    joint_table = joint_table.format(
        "{:.2f}", subset=joint_table.data.select_dtypes("number").columns
    )
    joint_table = joint_table.format("{:,}", subset=["Rank (Borda)"])
    joint_table = joint_table.highlight_min(
        subset=["Rank (Borda)"], props="font-weight: bold"
    )
    numerics = per_task.select_dtypes("number").columns
    per_task[numerics] = per_task[numerics].map(format_scores)
    per_task = per_task.style.highlight_max(
        subset=numerics, props="font-weight: bold"
    ).format("{:.2f}", subset=numerics)
    column_widths = get_column_widths(joint_table.data)
    # overriding for model name
    column_widths[1] = "250px"
    column_types = get_column_types(joint_table.data)
    # setting model name column to markdown
    column_types[1] = "markdown"
    return (
        gr.DataFrame(
            joint_table,
            column_widths=column_widths,
            datatype=column_types,
            wrap=True,
        ),
        gr.DataFrame(per_task),
    )
