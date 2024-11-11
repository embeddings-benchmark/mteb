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
    name, _ = name.split("]")
    return name[1:]


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
    df["Embedding Dimensions"] = df["Embedding Dimensions"].map(int)
    df["Max Tokens"] = df["Max Tokens"].map(int)
    df["Log(Tokens)"] = np.log10(df["Max Tokens"])
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
    )
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
        )
    )
    fig.update_traces(
        textposition="top center",
    )
    # fig.update_traces(marker=dict(size=14))
    fig.update_layout(
        font=dict(size=16, color="black"),
    )
    return fig
