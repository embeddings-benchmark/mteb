"""Helpers for lazy leaderboard plot updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


@dataclass(frozen=True)
class BenchmarkSelectOutputValues:
    """Values refreshed when the selected benchmark changes."""

    languages: list[str]
    domains: list[str]
    task_types: list[str]
    modalities: list[str]
    benchmark_tasks: list[str]
    scores: Any
    show_zero_shot: bool
    initial_models: list[str]
    display_radar: bool
    summary_raw: Any
    summary_table_value: Any
    per_task_table_value: Any
    per_language_table_value: Any
    language_tab_update: Any
    task_info_value: Any
    description_value: Any


def build_benchmark_select_outputs(
    values: BenchmarkSelectOutputValues,
    update: Callable[..., Any],
    plot_callables: tuple[Callable[[Any], Any], ...] = (),
) -> tuple[Any, ...]:
    """Build benchmark change outputs without computing plot figures."""
    _ = plot_callables
    return (
        update(choices=values.languages, value=values.languages),
        update(choices=values.domains, value=values.domains),
        update(choices=values.task_types, value=values.task_types),
        update(choices=values.modalities, value=values.modalities),
        update(choices=values.benchmark_tasks, value=values.benchmark_tasks),
        values.scores,
        update(visible=values.show_zero_shot),
        values.initial_models,
        update(visible=values.display_radar),
        values.summary_raw,
        values.summary_table_value,
        values.per_task_table_value,
        values.per_language_table_value,
        values.language_tab_update,
        values.task_info_value,
        values.description_value,
    )
