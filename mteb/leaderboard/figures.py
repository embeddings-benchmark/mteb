import logging
import re
from typing import get_args
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mteb
from mteb.abstasks.task_metadata import TaskType

logger = logging.getLogger(__name__)


def _text_plot(text: str):
    """Returns empty scatter plot with text added, this can be great for error messages."""
    return px.scatter(template="plotly_white").add_annotation(
        text=text, showarrow=False, font=dict(size=20)
    )


def _failsafe_plot(fun):
    """Decorator that turns the function producing a figure failsafe.

    This is necessary, because once a Callback encounters an exception it
    becomes useless in Gradio.

    Returns:
         A text plot with the error message if an exception occurs.
    """

    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return _text_plot(f"Couldn't produce plot. Reason: {e}")

    return wrapper


def _parse_n_params(params: float | None) -> int | float:
    """Specified in billions."""
    if params is None or np.isnan(params):
        return None
    return int(params * 1e9)


def _parse_model_name(name: str) -> str:
    if name is None:
        return ""
    if "]" not in name:
        return name
    name, _ = name.split("]")
    return name[1:]


def _parse_float(value) -> float:
    if value is None or np.isnan(value):
        return np.nan
    return float(value)


def _process_max_tokens(x):
    if pd.isna(x) or x is None or np.isinf(x):
        return "Unknown"
    return str(int(x))


def _parse_markdown_model_cell(model_cell: str) -> tuple[str, str | None]:
    if model_cell is None or pd.isna(model_cell):
        return "", None
    model_cell = str(model_cell).strip()
    match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", model_cell)
    if match is None:
        return model_cell, None
    return match.group(1), match.group(2)


def _extract_hf_model_name(model_url: str | None) -> str | None:
    if model_url is None or "huggingface" not in model_url:
        return None
    parsed = urlparse(model_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        return None
    return f"{path_parts[0]}/{path_parts[1]}"


def _extract_model_name_and_release_date(model_cell: str) -> tuple[str, str | None]:
    display_name, model_url = _parse_markdown_model_cell(model_cell)
    model_name = _extract_hf_model_name(model_url)
    model_metas = {meta.name: meta for meta in mteb.get_model_metas()}

    model_meta = model_metas.get(model_name) if model_name else None
    release_date = model_meta.release_date if model_meta is not None else None
    return display_name, release_date


models_to_annotate = [
    "all-MiniLM-L6-v2",
    "clap-htsat-fused",
    "e5-v",
    "EVA02-CLIP-bigE-14-plus",
    "GritLM-7B",
    "LaBSE",
    "larger_clap_general",
    "LCO-Embedding-Omni-3B",
    "LCO-Embedding-Omni-7B",
    "MuQ-MuLan-large",
    "multilingual-e5-large-instruct",
    "Qwen2-Audio-7B",
    "VLM2Vec-Full",
    "voyage-multimodal-3",
    "wav2clip",
    "wav2vec2-xls-r-1b",
    "wavlm-base-plus-svmsclap-2023",
    "whisper-large-v3",
    "whisper-medium",
    "yamnet",
]


def _add_size_guide(fig: go.Figure):
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


@_failsafe_plot
def _performance_size_plot(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Number of Parameters"] = df["Number of Parameters (B)"].map(_parse_n_params)
    df["Model"] = df["Model"].map(_parse_model_name)
    df["model_text"] = df["Model"].where(df["Model"].isin(models_to_annotate), "")
    df["Embedding Dimensions"] = df["Embedding Dimensions"].map(_parse_float)
    df["Max Tokens"] = df["Max Tokens"].map(_parse_float)
    df["Log(Tokens)"] = np.log10(df["Max Tokens"])
    df["Mean (Task)"] = df["Mean (Task)"].map(_parse_float)
    df = df.dropna(
        subset=["Mean (Task)", "Number of Parameters", "Embedding Dimensions"]
    )
    if not len(df.index):
        return go.Figure()
    min_score, max_score = df["Mean (Task)"].min(), df["Mean (Task)"].max()
    df["sqrt(dim)"] = np.sqrt(df["Embedding Dimensions"])
    df["Max Tokens"] = df["Max Tokens"].apply(lambda x: _process_max_tokens(x))
    rank_column = "Rank (Borda)" if "Rank (Borda)" in df.columns else "Rank (Mean Task)"
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
            rank_column: True,
            "Log(Tokens)": False,
            "sqrt(dim)": False,
            "model_text": False,
        },
        hover_name="Model",
        color_continuous_scale=px.colors.sequential.Greens,
    )
    # Note: it's important that this comes before setting the size mode
    fig = _add_size_guide(fig)
    fig.update_traces(
        marker=dict(
            sizemode="diameter",
            sizeref=1.5,
            sizemin=0,
        )
    )
    fig.add_annotation(x=1e9, y=10, text="Model size:")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Max Tokens",
            tickvals=[2, 3, 4, 5],
            ticktext=[
                "100",
                "1K",
                "10K",
                "100K",
            ],
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
    )
    fig.update_traces(
        textposition="top center",
    )
    fig.update_layout(
        font=dict(size=16, color="black"),
        margin=dict(b=20, t=10, l=20, r=10),
    )
    return fig


@_failsafe_plot
def _performance_over_time_plot(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    score_column = "Mean (Task)"
    if score_column not in df.columns or "Model" not in df.columns:
        return _text_plot(
            "Couldn't produce timeline plot. Required columns are missing."
        )

    model_release_info = df["Model"].map(_extract_model_name_and_release_date)
    df["Model"] = model_release_info.map(lambda x: x[0])
    df["Release Date"] = model_release_info.map(lambda x: x[1])
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df[score_column] = df[score_column].map(_parse_float)

    df = df.dropna(subset=["Release Date", score_column]).sort_values(
        ["Release Date", score_column], ascending=[True, False]
    )
    if not len(df.index):
        return _text_plot(
            "Couldn't produce timeline plot. No models have a valid release date and score."
        )

    df["score"] = df[score_column].cummax()
    fig = px.scatter(
        df,
        x="Release Date",
        y=score_column,
        template="plotly_white",
        hover_name="Model",
        hover_data={
            "Release Date": "|%Y-%m-%d",
            score_column: True,
            "score": False,
        },
    )

    fig.add_trace(
        go.Scatter(
            x=df["Release Date"],
            y=df["score"],
            mode="lines",
            line=dict(color="#1f7a1f", width=2, shape="hv"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        )
    )

    fig.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title="Release Date",
        yaxis_title=f"{score_column} score",
        showlegend=False,
        font=dict(size=16, color="black"),
        margin=dict(b=20, t=10, l=20, r=10),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
    )
    return fig


TOP_N = 5
task_types = sorted(get_args(TaskType))
task_types.remove("InstructionRetrieval")
# Not displayed, because the scores are negative,
# doesn't work well with the radar chart.

# Create a mapping for task types that lose digits when processed by _split_on_capital
# e.g., "Any2AnyRetrieval" -> "Any Any Retrieval" -> "AnyAnyRetrieval" (loses the "2")
_task_type_normalized = {t: "".join(t.split()) for t in task_types}
# Add reverse mappings for task types with digits that get lost
# "AnyAnyRetrieval" should also match to "Any2AnyRetrieval"
_task_type_aliases = {
    "AnyAnyRetrieval": "Any2AnyRetrieval",
    "AnyAnyMultilingualRetrieval": "Any2AnyMultilingualRetrieval",
}

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


def _is_task_type_column(column: str) -> bool:
    """Check if a column name corresponds to a task type.

    Handles cases where task types with digits (e.g., Any2AnyRetrieval) become
    column names without digits (e.g., "Any Any Retrieval") after _split_on_capital.
    """
    normalized = "".join(column.split())
    if normalized in task_types:
        return True
    # Check aliases for task types that lose digits
    if normalized in _task_type_aliases:
        return True
    return False


@_failsafe_plot
def _radar_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["Model"] = df["Model"].map(_parse_model_name)
    # Remove whitespace and match to task types
    task_type_columns = [
        column for column in df.columns if _is_task_type_column(column)
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
        font=dict(size=13, color="black"),
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
