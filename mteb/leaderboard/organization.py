"""Display-only organization groups for an already filtered benchmark summary."""

from __future__ import annotations

import math
from html import escape
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from mteb.benchmarks._create_table import SummaryTable


def _organization(model: str) -> str:
    # Model identifiers use the Hub's organization/model convention. Do not
    # merge unrelated identifiers without an organization into one group.
    organization, separator, _ = model.partition("/")
    return organization if separator and organization else model


def group_summary_by_organization(
    summary: SummaryTable,
) -> list[tuple[str, pd.DataFrame]]:
    """Order organizations and their models by the benchmark's primary metric.

    The first row is the representative's *whole* row, not column-wise maxima.
    Missing scores sort last; ties use the canonical model name, then original
    row order (which also preserves distinct experiment variants).
    """
    if summary.is_empty or summary.df.is_empty():
        return []
    frame = summary.df.to_pandas()
    frame = frame.sort_values(
        [summary.primary_metric_col, "Model"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    )
    return list(frame.groupby(frame["Model"].map(_organization), sort=False))


def _cell_text(column: str, value: object) -> str:
    if value is None or (isinstance(value, Real) and not math.isfinite(value)):
        return "—"
    if isinstance(value, Real):
        if column == "Zero-shot":
            return "⚠️ NA" if value == -1 else f"{value:.0f}%"
        if column.startswith("Rank") or column in {
            "Embedding Dimensions",
            "Max Tokens",
        }:
            return f"{value:.0f}"
        if column in {"Total Parameters (B)", "Active Parameters (B)"}:
            return f"{value:.1f}" if value >= 1 else f"{value:.3f}"
        return f"{value * 100:.2f}"
    return str(value)


def organization_summary_html(summary: SummaryTable) -> str:
    """Render keyboard-expandable rows without adding a frontend dependency.

    All result/metadata text is escaped. Native details/summary elements keep
    expansion client-side; changes to the filtered summary reset the groups.
    """
    groups = group_summary_by_organization(summary)
    if not groups:
        return "<p>No results for the selected filters.</p>"
    leading = ["Model", summary.primary_metric_col]
    columns = leading + [
        column
        for column in summary.df.columns
        if column not in {*leading, "Release Date"}
    ]
    widths = ["minmax(16rem, 2fr)" if c == "Model" else "8rem" for c in columns]
    grid = "minmax(12rem, 1.5fr) " + " ".join(widths)

    def row(label: str, values: list[object], *, header: bool = False) -> str:
        cells = [f'<span class="org-label">{escape(label)}</span>']
        for column, value in zip(columns, values, strict=True):
            text = str(value) if header else _cell_text(column, value)
            cells.append(f'<span title="{escape(column)}">{escape(text)}</span>')
        return '<span class="org-row">' + "".join(cells) + "</span>"

    parts = [
        '<div class="org-groups">',
        f"""<style>
        .org-groups {{ overflow-x: auto; color: var(--body-text-color); }}
        .org-groups .org-row {{ display: grid; grid-template-columns: {grid};
            align-items: center; min-width: max-content; }}
        .org-groups .org-row > span {{ padding: .6rem; overflow-wrap: anywhere; }}
        .org-groups .org-header {{ font-weight: 600; }}
        .org-groups details {{ width: max-content; min-width: 100%;
            border-bottom: 1px solid var(--border-color-primary, #ddd); }}
        .org-groups summary {{ cursor: pointer; list-style: none; }}
        .org-groups summary .org-label::before {{ content: '▸ '; }}
        .org-groups details[open] > summary .org-label::before {{ content: '▾ '; }}
        .org-groups summary:focus-visible {{ outline: 2px solid var(--color-accent, #2563eb);
            outline-offset: -2px; }}
        .org-groups .org-members {{ background: var(--background-fill-secondary, #f5f5f5); }}
        </style>""",
        (
            "<p>Each organization shows its highest-scoring eligible model by "
            f"{escape(summary.primary_metric_col)}. Expand to see all its models. "
            "Ranks refer to individual models; downloads and plots remain ungrouped.</p>"
        ),
        '<div class="org-header">'
        + row("Organization", columns, header=True)
        + "</div>",
    ]
    for organization, models in groups:
        values = models[columns].to_numpy().tolist()
        parts.extend(
            [
                "<details><summary>",
                row(f"{organization} ({len(models)})", values[0]),
                '</summary><div class="org-members">',
                *(row("", model) for model in values),
                "</div></details>",
            ]
        )
    parts.append("</div>")
    return "".join(parts)
