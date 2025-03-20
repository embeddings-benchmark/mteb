from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mteb.abstasks.TaskMetadata import TASK_TYPE


def text_plot(text: str):
    """Returns empty scatter plot with text added, this can be great for error messages."""
    return px.scatter(template="plotly_white").add_annotation(
        text=text, showarrow=False, font=dict(size=20)
    )


def failsafe_plot(fun):
    """Decorator that turns the function producing a figure failsafe.
    This is necessary, because once a Callback encounters an exception it
    becomes useless in Gradio.
    """

    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            return text_plot(f"Couldn't produce plot. Reason: {e}")

    return wrapper


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
        if value == "Infinite":
            return np.inf
        else:
            return float(value)
    except ValueError:
        return np.nan


def process_max_tokens(x):
    if pd.isna(x):
        return "Unknown"
    if np.isinf(x):
        return "Infinite"
    return str(int(x))


models_to_annotate = [
    "all-MiniLM-L6-v2",
    "GritLM-7B",
    "LaBSE",
    "multilingual-e5-large-instruct",
    "EVA02-CLIP-bigE-14-plus",
    "voyage-multimodal-3",
    "e5-v",
    "VLM2Vec-Full",
]


def add_size_guide(fig: go.Figure):
    xpos = [2 * 1e6] * 4
    ypos = [7.8, 8.5, 9, 10]
    sizes = [256, 1024, 2048, 4096]
    fig.add_trace(
        go.Scatter(
            showlegend=False,
            opacity=0.3,
            mode="markers",
            marker=dict(
                size=np.sqrt(sizes),
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2),
            ),
            x=xpos,
            y=ypos,
        )
    )
    fig.add_annotation(
        text="<b>Embedding Size</b>",
        font=dict(size=16),
        x=np.log10(10 * 1e6),
        y=10,
        showarrow=False,
        opacity=0.3,
    )
    return fig


@failsafe_plot
def performance_size_plot(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Number of Parameters"] = df["Number of Parameters"].map(parse_n_params)
    df["Model"] = df["Model"].map(parse_model_name)
    df["model_text"] = df["Model"].where(df["Model"].isin(models_to_annotate), "")
    df["Embedding Dimensions"] = df["Embedding Dimensions"].map(parse_float)
    df["Max Tokens"] = df["Max Tokens"].map(parse_float)
    df["Log(Tokens)"] = np.log10(df["Max Tokens"])
    df["Mean (Task)"] = df["Mean (Task)"].map(parse_float)
    df = df.dropna(
        subset=["Mean (Task)", "Number of Parameters", "Embedding Dimensions"]
    )
    if not len(df.index):
        return go.Figure()
    min_score, max_score = df["Mean (Task)"].min(), df["Mean (Task)"].max()
    df["sqrt(dim)"] = np.sqrt(df["Embedding Dimensions"])
    df["Max Tokens"] = df["Max Tokens"].apply(lambda x: process_max_tokens(x))
    fig = px.scatter(
        df,
        x="Number of Parameters",
        y="Mean (Task)",
        log_x=True,
        template="plotly_white",
        text="model_text",
        size="sqrt(dim)",
        color="Log(Tokens)",
        range_color=[2, 5],
        range_y=[min(0, min_score * 1.25), max_score * 1.25],
        hover_data={
            "Max Tokens": True,
            "Embedding Dimensions": True,
            "Number of Parameters": True,
            "Mean (Task)": True,
            "Rank (Borda)": True,
            "Log(Tokens)": False,
            "sqrt(dim)": False,
            "model_text": False,
        },
        hover_name="Model",
    )
    # Note: it's important that this comes before setting the size mode
    fig = add_size_guide(fig)
    fig.update_traces(
        marker=dict(
            sizemode="diameter",
            sizeref=1.5,
            sizemin=0,
        )
    )
    fig.add_annotation(x=1e9, y=10, text="Model size:")
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
task_types = sorted(get_args(TASK_TYPE))
task_types.remove("InstructionRetrieval")
# Not displayed, because the scores are negative,
# doesn't work well with the radar chart.

line_colors = [
    "#EE4266",
    "#00a6ed",
    "#ECA72C",
    "#B42318",
    "#3CBBB1",
]
fill_colors = [
    "rgba(238,66,102,0.05)",
    "rgba(0,166,237,0.05)",
    "rgba(236,167,44,0.05)",
    "rgba(180,35,24,0.05)",
    "rgba(60,187,177,0.05)",
]


@failsafe_plot
def radar_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Model"] = df["Model"].map(parse_model_name)
    # Remove whitespace
    task_type_columns = [
        column for column in df.columns if "".join(column.split()) in task_types
    ]
    if len(task_type_columns) <= 1:
        raise ValueError(
            "Couldn't produce radar chart, the benchmark only contains one task category."
        )
    df = df[["Model", *task_type_columns]].set_index("Model")
    df = df.mask(df == "", np.nan)
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
                fillcolor="rgba(0,0,0,0)",
            )
        )
    fig.update_layout(
        font=dict(size=13, color="black"),  # noqa
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
            y=-0.35,
            xanchor="center",
            x=0.4,
            itemwidth=30,
            font=dict(size=13),
            entrywidth=0.6,
            entrywidthmode="fraction",
        ),
        margin=dict(l=0, r=16, t=30, b=30),
        autosize=True,
    )
    return fig
