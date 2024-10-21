from __future__ import annotations

import numpy as np
import pandas as pd

from mteb.overview import get_task


def scores_to_table(scores_long: list[dict]):
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
    overall_mean = (
        data.groupby(["model_name", "model_revision"])[["score"]]
        .agg(np.nanmean)
        .rename(columns={"score": "mean"})
    )
    joint_table = overall_mean.join([typed_mean, mean_per_type, per_task]).reset_index()
    joint_table = joint_table.sort_values("mean", ascending=False)
    joint_table = joint_table.rename(
        columns={
            "model_name": "Model",
            "mean_by_task_type": "Mean by Task Type",
            "mean": "Mean",
        }
    )
    joint_table = joint_table.drop(columns=["model_revision"])
    return joint_table
