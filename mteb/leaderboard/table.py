from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd

from mteb.overview import get_task


def format_scores(score: float) -> float:
    return score * 100


def scores_to_tables(scores_long: list[dict]):
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
    per_task = data.pivot(
        index=["model_name", "model_revision"], columns="task_name", values="score"
    )
    to_remove = per_task.isna().any(axis="columns")
    overall_mean = (
        data.groupby(["model_name", "model_revision"])[["score"]]
        .agg(np.nanmean)
        .rename(columns={"score": "mean"})
    )
    per_task = per_task[~to_remove]
    mean_per_type = mean_per_type[~to_remove]
    overall_mean = overall_mean[~to_remove]
    mean_rank = per_task.rank(ascending=False, numeric_only=True).mean(
        axis=1, skipna=True
    )
    joint_table = overall_mean.join([typed_mean, mean_per_type])
    joint_table.insert(0, "mean_rank", mean_rank)
    joint_table = joint_table.reset_index()
    joint_table = joint_table.sort_values("mean", ascending=False)
    joint_table["model_name"] = joint_table["model_name"].map(
        lambda name: name.split("/")[-1]
    )
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": "Mean by Task Type",
            "mean": "Mean",
            "mean_rank": "Mean Rank",
        }
    )
    joint_table = joint_table.drop(columns=["model_revision"])
    joint_table.insert(
        0, "Rank", joint_table["Mean"].rank(ascending=False).map(int).map(str)
    )
    per_task = per_task.rename(
        columns={
            "model_name": "Model",
        }
    )
    per_task = per_task.reset_index().drop(columns=["model_revision"])
    numerics = joint_table.select_dtypes("number").columns
    to_format = ["Mean", "Mean by Task Type", *mean_per_type.columns]
    joint_table[to_format] = joint_table[to_format].map(format_scores)
    joint_table = joint_table.style.highlight_max(
        subset=to_format,
        props="font-weight: bold",
    ).format("{:.2f}", subset=numerics)
    joint_table = joint_table.highlight_min(
        subset=["Mean Rank"], props="font-weight: bold"
    )
    numerics = per_task.select_dtypes("number").columns
    per_task[numerics] = per_task[numerics].map(format_scores)
    per_task = per_task.style.highlight_max(
        subset=numerics, props="font-weight: bold"
    ).format("{:.2f}", subset=numerics)
    return joint_table, per_task
