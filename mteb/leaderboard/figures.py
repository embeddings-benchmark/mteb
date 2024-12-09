from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def parse_n_params(text: str) -> int:
    if text.endswith("M"):
        return float(text[:-1]) * 1e6
    if text.endswith("B"):
        return float(text[:-1]) * 1e9


def parse_model_name(name: str) -> str:
    if name is None:
        return ""
    if "]" not in name:
        return name
    name, _ = name.split("]")
    return name[1:]


def parse_float(value) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan


models_to_annotate = [
    "all-MiniLM-L6-v2",
    "GritLM-7B",
    "LaBSE",
    "multilingual-e5-large-instruct",
]


def performance_size_plot(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Number of Parameters"] = df["Number of Parameters"].map(parse_n_params)
    df["Model"] = df["Model"].map(parse_model_name)
    df["model_text"] = df["Model"].where(df["Model"].isin(models_to_annotate), "")
    df["Embedding Dimensions"] = df["Embedding Dimensions"].map(parse_float)
    df["Max Tokens"] = df["Max Tokens"].map(parse_float)
    df["Log(Tokens)"] = np.log10(df["Max Tokens"])
    df["Mean (Task)"] = df["Mean (Task)"].map(parse_float)
    df = df.dropna(subset=["Mean (Task)", "Number of Parameters"])
    if not len(df.index):
        return go.Figure()
    min_score, max_score = df["Mean (Task)"].min(), df["Mean (Task)"].max()
    fig = px.scatter(
        df,
        x="Number of Parameters",
        y="Mean (Task)",
        log_x=True,
        template="plotly_white",
        text="model_text",
        size="Embedding Dimensions",
        color="Log(Tokens)",
        range_color=[2, 5],
        range_x=[8 * 1e6, 11 * 1e9],
        range_y=[min(0, min_score * 1.25), max_score * 1.25],
        hover_data={
            "Max Tokens": True,
            "Embedding Dimensions": True,
            "Number of Parameters": True,
            "Mean (Task)": True,
            "Rank (Borda)": True,
            "Log(Tokens)": False,
            "model_text": False,
        },
        hover_name="Model",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(  # noqa
            title="Max Tokens",
            tickvals=[2, 3, 4, 5],
            ticktext=[
                "100",
                "1K",
                "10K",
                "100K",
            ],
        ),
        hoverlabel=dict(  # noqa
            bgcolor="white",
            font_size=16,
        ),
    )
    fig.update_traces(
        textposition="top center",
    )
    fig.update_layout(
        font=dict(size=16, color="black"),  # noqa
        margin=dict(b=20, t=10, l=20, r=10),  # noqa
    )
    return fig


TOP_N = 5
task_types = [
    "BitextMining",
    "Classification",
    "MultilabelClassification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
    # "InstructionRetrieval",
    # Not displayed, because the scores are negative,
    # doesn't work well with the radar chart.
    "Speed",
]

line_colors = [
    "#EE4266",
    "#00a6ed",
    "#ECA72C",
    "#B42318",
    "#3CBBB1",
]
fill_colors = [
    "rgba(238,66,102,0.2)",
    "rgba(0,166,237,0.2)",
    "rgba(236,167,44,0.2)",
    "rgba(180,35,24,0.2)",
    "rgba(60,187,177,0.2)",
]


def radar_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Model"] = df["Model"].map(parse_model_name)
    # Remove whitespace
    task_type_columns = [
        column for column in df.columns if "".join(column.split()) in task_types
    ]
    df = df[["Model", *task_type_columns]].set_index("Model")
    df = df.replace("", np.nan)
    df = df.dropna()
    df = df.head(TOP_N)
    df = df.iloc[::-1]
    fig = go.Figure()
    for i, (model_name, row) in enumerate(df.iterrows()):
        fig.add_trace(
            go.Scatterpolar(
                name=model_name,
                r=[row[task_type] for task_type in task_type_columns]
                + [row[task_type_columns[0]]],
                theta=task_type_columns + [task_type_columns[0]],
                showlegend=True,
                mode="lines",
                line=dict(width=2, color=line_colors[i]),
                fill="toself",
                fillcolor=fill_colors[i],
            )
        )
    fig.update_layout(
        font=dict(size=16, color="black"),  # noqa
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor="black",
                linecolor="rgba(0,0,0,0)",
                gridwidth=1,
                showticklabels=False,
                ticks="",
            ),
            angularaxis=dict(
                gridcolor="black", gridwidth=1.5, linecolor="rgba(0,0,0,0)"
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
            xanchor="left",
            x=-0.05,
            entrywidthmode="fraction",
            entrywidth=1 / 5,
        ),
    )
    return fig
