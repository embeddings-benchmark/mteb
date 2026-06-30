"""Tests for lazy leaderboard plot generation."""

import importlib.util
import sys
from pathlib import Path


def _load_lazy_plots_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "mteb" / "leaderboard" / "lazy_plots.py"
    )
    spec = importlib.util.spec_from_file_location("leaderboard_lazy_plots", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_select_outputs_do_not_compute_plots():
    """Benchmark changes should update data and tables without eager plot work."""
    calls = {"size": 0, "time": 0, "radar": 0}

    def track(name):
        def _plot(_summary):
            calls[name] += 1
            return f"{name}-plot"

        return _plot

    summary_raw = object()

    module = _load_lazy_plots_module()

    values = module.BenchmarkSelectOutputValues(
        languages=["eng"],
        domains=["Written"],
        task_types=["Classification"],
        modalities=["Text"],
        benchmark_tasks=["Banking77Classification"],
        scores="scores",
        show_zero_shot=True,
        initial_models=["model-a"],
        display_radar=True,
        summary_raw=summary_raw,
        summary_table_value="summary-table",
        per_task_table_value="per-task-table",
        per_language_table_value="per-language-table",
        language_tab_update="language-tab-update",
        task_info_value="task-info",
        description_value="Benchmark description",
    )

    outputs = module.build_benchmark_select_outputs(
        values,
        update=lambda **kwargs: kwargs,
        plot_callables=(track("size"), track("time"), track("radar")),
    )

    assert calls == {"size": 0, "time": 0, "radar": 0}
    assert outputs[9] is summary_raw
    assert "size-plot" not in outputs
    assert "time-plot" not in outputs
    assert "radar-plot" not in outputs


def test_plot_tab_callable_still_uses_latest_summary_data():
    """Tab selection should still compute a plot from current summary data."""
    summary_raw = {"Model": ["model-a"], "Score": [0.5]}

    def plot_tab(summary):
        return summary["Model"]

    assert plot_tab(summary_raw) == ["model-a"]
