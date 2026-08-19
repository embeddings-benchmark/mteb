from __future__ import annotations

import pandas as pd

from mteb.leaderboard.figures import _performance_size_plot


def test_performance_size_plot_includes_optional_total_parameters_in_hover():
    summary = pd.DataFrame(
        {
            "Model": ["org/model-a", "org/model-b"],
            "Active Parameters (B)": [0.6, 1.2],
            "Total Parameters (B)": [0.8, None],
            "Embedding Dimensions": [768, 1024],
            "Max Tokens": [512, 2048],
            "Mean (Task)": [0.55, 0.62],
            "Rank (Borda)": [2, 1],
        }
    )

    figure = _performance_size_plot(summary)

    assert len(figure.data) == 1
    trace = figure.data[0]
    assert list(trace.x) == [600_000_000, 1_200_000_000]
    assert "Number of Total Parameters=%{customdata[3]}" in trace.hovertemplate
    assert trace.customdata[0][3] == 800_000_000
    assert pd.isna(trace.customdata[1][3])
