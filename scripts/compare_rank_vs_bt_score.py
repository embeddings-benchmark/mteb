"""Compare BT score with the leaderboard's rank, Mean (Task) and Mean (TaskType).

Builds each benchmark's summary in-process with the same code the API uses
(``mteb.api.aggregators.build_benchmark_summary`` over the local ``ResultCache`` /
cached results frames), so no running server is needed:

    python scripts/compare_rank_vs_bt_score.py
    python scripts/compare_rank_vs_bt_score.py --benchmark "MTEB(eng, v2)" --benchmark LongEmbed

For every benchmark, BT score is compared against each reference ordering that applies:
  * ``Rank``: the summary's ``rank`` (always);
  * ``Mean (Task)`` / ``Mean (TaskType)``: only if the benchmark declares that
    aggregation and at least three rated models have a value (models missing a task
    have a null mean and drop out of that comparison).

Writes:
  * ``--summary-out`` (default ``rank_vs_bt_score_summary.csv``): ONE table, one row per
    benchmark, with Spearman / Kendall correlation, top-10 overlap, whether top 1
    agrees, the number of models compared and the biggest mover against each
    reference. A markdown copy (compact: no mover columns) goes next to it
    (``.md``) and is printed to stdout. ``-`` means not applicable.
  * ``--positions-out`` (default ``rank_vs_bt_score_positions.csv``): one row per
    (benchmark, model, experiment) with the scores and every position. Positions
    are 1-based within a benchmark; mean positions are among models that have that
    mean.

Pass ``--xlsx-out FILE.xlsx`` (needs openpyxl) for an Excel workbook with a summary
sheet and one sheet per benchmark: every model's position under each ranking system
(Rank, Mean (Task), Mean (TaskType), BT score) and the changes between them.

Pass ``--plots-dir DIR`` (needs matplotlib, in the ``leaderboard`` extra) to also
write ``overview.png`` (Spearman per benchmark) and one ``<benchmark>.png`` each with
a rank-vs-BT score position scatter and the top models' BT score with 95% bootstrap intervals.

Rows without a ``bt_score`` are skipped, as are benchmarks with fewer than three rated
rows.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scipy.stats import kendalltau, rankdata, spearmanr

import mteb
from mteb.benchmarks.benchmark import BenchmarkAggregation
from mteb.api.aggregators import _extract_variant_kwargs, build_benchmark_summary
from mteb.api.frames import _load_per_benchmark_frames, get_cache

if TYPE_CHECKING:
    from mteb.cache.result_cache import ResultCache

TOP_K = 10
MIN_ROWS = 3
TOP_N_PLOT = 30

# (key, label, aggregation that must be declared, row field, higher-is-better).
# ``rank`` is lower-is-better and always available.
REFERENCES: tuple[tuple[str, str, str | None, str, bool], ...] = (
    ("rank", "Rank", None, "rank", False),
    ("borda", "Borda", None, "borda_rank", False),
    ("mean_task", "Mean (Task)", "mean_task", "mean_task", True),
    ("mean_task_type", "Mean (TaskType)", "mean_task_type", "mean_task_type", True),
    ("aggregated", "Aggregated", None, "aggregated_score", False),
)


def _main_ranking(name: str) -> str:
    """What the leaderboard's ``rank`` column is based on for this benchmark.

    Same rule as ``Benchmark._create_summary_table``: the benchmark's
    ``summary_sort_column`` (one column, or a tie-break chain) if it sets one;
    otherwise Borda, or Mean (Subset) for subset-weighted benchmarks.
    """
    bench = mteb.get_benchmark(name)
    sort = bench.summary_sort_column
    if sort:
        return sort if isinstance(sort, str) else ", then ".join(sort)
    if BenchmarkAggregation.MEAN_SUBSET in bench.aggregations:
        return "Mean (Subset)"
    return "Borda"


def _experiment_key(model: str, experiments: dict[str, Any] | None) -> tuple[str, str]:
    return model, json.dumps(
        experiments, sort_keys=True, default=str
    ) if experiments else ""


def _borda_ranks(name: str) -> dict[tuple[str, str], int]:
    """The leaderboard's own ``Rank (Borda)`` per (model, experiment).

    Read from the summary table mteb builds, since rank itself is not always Borda
    (e.g. MIEB ranks by Mean (TaskType)) and Borda is computed over every model in
    the results frame, including ones the API does not list.
    """
    bench = mteb.get_benchmark(name)
    frames, _ = _load_per_benchmark_frames()
    long_df = frames.get(bench.name)
    if long_df is None or long_df.is_empty():
        return {}
    variants = _extract_variant_kwargs(long_df)
    table = bench._create_summary_table(long_df).df
    return {
        _experiment_key(
            row["Model"],
            variants.get((row["Model"], row.get("_experiment_id") or ""))
            if row.get("_experiment_id")
            else None,
        ): int(row["Rank (Borda)"])
        for row in table.iter_rows(named=True)
        if row.get("Rank (Borda)") is not None
    }


def _load_summary(name: str, cache: ResultCache) -> dict[str, Any]:
    """Build one benchmark's summary and reduce it to the fields compared here."""
    summary = asyncio.run(build_benchmark_summary(name, cache))
    borda = _borda_ranks(name)
    return {
        "main_ranking": _main_ranking(name),
        "tasks": summary.tasks,
        "aggregations": list(summary.aggregations),
        "rows": [
            {
                "rank": r.rank,
                "model": r.model.name,
                "experiments": r.experiments,
                "mean_task": r.mean_task,
                "mean_task_type": r.mean_task_type,
                "borda_rank": borda.get(_experiment_key(r.model.name, r.experiments)),
                "bt_score": r.bt_score,
                "bt_score_low": r.bt_score_low,
                "bt_score_high": r.bt_score_high,
            }
            for r in summary.rows
        ],
    }


def _positions(
    rows: list[dict[str, Any]], field: str, higher_better: bool
) -> dict[int, int]:
    """1-based position of each row (by ``id``) when ordered best-first by ``field``."""
    ordered = sorted(rows, key=lambda r: -r[field] if higher_better else r[field])
    return {id(r): i + 1 for i, r in enumerate(ordered)}


def _agreement(
    rows: list[dict[str, Any]], field: str, higher_better: bool
) -> dict[str, Any] | None:
    """How BT score's ordering agrees with ``field``'s, over rows that have both."""
    rows = [r for r in rows if r.get(field) is not None]
    if len(rows) < MIN_ROWS:
        return None
    ref_pos = _positions(rows, field, higher_better)
    bt_pos = _positions(rows, "bt_score", True)
    ref_score = [r[field] if higher_better else -r[field] for r in rows]
    bt_values = [r["bt_score"] for r in rows]
    mover = max(rows, key=lambda r: abs(ref_pos[id(r)] - bt_pos[id(r)]))
    return {
        "n": len(rows),
        "spearman": float(spearmanr(ref_score, bt_values)[0]),
        "kendall": float(kendalltau(ref_score, bt_values)[0]),
        "top_k_overlap": sum(
            ref_pos[id(r)] <= TOP_K and bt_pos[id(r)] <= TOP_K for r in rows
        ),
        "same_first": min(rows, key=lambda r: ref_pos[id(r)])
        is min(rows, key=lambda r: bt_pos[id(r)]),
        "mover": f"{mover['model']} (reference {ref_pos[id(mover)]} → BT score {bt_pos[id(mover)]})",
        "positions": ref_pos,
    }


def _aggregate(
    rated: list[dict[str, Any]], mean_fields: list[str]
) -> dict[int, dict[str, float]]:
    """Average position across the ranking systems, over all rated models.

    Systems: Borda, BT score and each applicable mean. Positions are fractional (ties
    share their average position). A model without a mean sits in the shared tail
    position of that system, after every model that has one, as the leaderboard puts
    missing values last. Ordering by the average equals a Borda count over the
    rankings.
    """
    n_total = len(rated)
    inputs: dict[str, list[float]] = {
        "agg_bt_score": list(
            rankdata([-r["bt_score"] for r in rated], method="average")
        ),
    }
    # Borda is a rank (lower is better, ties already share the best rank); the
    # means are scores (higher is better). A row without a value takes the tail.
    for field in ("borda_rank", *mean_fields):
        sign = 1 if field == "borda_rank" else -1
        have = [r[field] for r in rated if r[field] is not None]
        ranks = iter(rankdata([sign * v for v in have], method="average"))
        tail = (len(have) + 1 + n_total) / 2
        inputs["agg_borda" if field == "borda_rank" else f"agg_{field}"] = [
            float(next(ranks)) if r[field] is not None else tail for r in rated
        ]
    return {
        id(r): {
            **{k: v[i] for k, v in inputs.items()},
            "aggregated_score": sum(v[i] for v in inputs.values()) / len(inputs),
        }
        for i, r in enumerate(rated)
    }


def _compare(
    name: str, summary: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """One summary-table row for the benchmark plus its per-model position rows."""
    rows = [r for r in summary["rows"] if r.get("bt_score") is not None]
    if len(rows) < MIN_ROWS:
        return None

    mean_fields = [
        field
        for _key, _label, aggregation, field, _hb in REFERENCES
        if aggregation is not None
        and aggregation in summary["aggregations"]
        and sum(r[field] is not None for r in rows) >= MIN_ROWS
    ]
    aggregated = _aggregate(rows, mean_fields)
    agg_order = rankdata(
        [aggregated[id(r)]["aggregated_score"] for r in rows], method="min"
    )
    for r, agg_position in zip(rows, agg_order, strict=True):
        r.update(aggregated[id(r)])
        r["aggregated_position"] = int(agg_position)

    stats: dict[str, Any] = {
        "benchmark": name,
        "main_ranking": summary["main_ranking"],
        "models": len(rows),
        "tasks": len(summary["tasks"]),
    }
    ref_positions: dict[str, dict[int, int]] = {}
    for key, _label, aggregation, field, higher_better in REFERENCES:
        result = (
            _agreement(rows, field, higher_better)
            if aggregation is None or aggregation in summary["aggregations"]
            else None
        )
        if result is None:
            continue
        ref_positions[key] = result.pop("positions")
        stats.update({f"{key}_{k}": v for k, v in result.items()})

    bt_pos = _positions(rows, "bt_score", True)
    # Mean positions are among the models that have that mean, so compare them
    # with BT score positions among the same models.
    bt_pos_in: dict[str, dict[int, int]] = {
        key: _positions([r for r in rows if r[key] is not None], "bt_score", True)
        for key in ("mean_task", "mean_task_type")
        if key in ref_positions
    }
    positions = [
        {
            "benchmark": name,
            "main_ranking": summary["main_ranking"],
            "model": r["model"],
            "experiment": json.dumps(r["experiments"]) if r["experiments"] else "",
            # The leaderboard's rank as shown (tied models share a rank).
            "rank_position": r["rank"],
            "mean_task_position": ref_positions.get("mean_task", {}).get(id(r)),
            "mean_task_type_position": ref_positions.get("mean_task_type", {}).get(
                id(r)
            ),
            "bt_score_position": bt_pos[id(r)],
            "bt_score_vs_rank_delta": r["rank"] - bt_pos[id(r)],
            "borda_position": r["borda_rank"],
            "aggregated_position": r["aggregated_position"],
            "aggregated_score": round(r["aggregated_score"], 2),
            "bt_score_vs_aggregated_delta": r["aggregated_position"] - bt_pos[id(r)],
            "bt_score_vs_borda_delta": (
                None if r["borda_rank"] is None else r["borda_rank"] - bt_pos[id(r)]
            ),
            **{
                k: r.get(k)
                for k in (
                    "agg_borda",
                    "agg_bt_score",
                    "agg_mean_task",
                    "agg_mean_task_type",
                )
            },
            "delta_vs_mean_task": _delta(ref_positions, bt_pos_in, "mean_task", r),
            "delta_vs_mean_task_type": _delta(
                ref_positions, bt_pos_in, "mean_task_type", r
            ),
            "mean_task": r["mean_task"],
            "mean_task_type": r["mean_task_type"],
            "bt_score": round(r["bt_score"], 1),
            "bt_score_low": None
            if r["bt_score_low"] is None
            else round(r["bt_score_low"], 1),
            "bt_score_high": None
            if r["bt_score_high"] is None
            else round(r["bt_score_high"], 1),
        }
        for r in sorted(rows, key=lambda r: ref_positions["rank"][id(r)])
    ]
    return stats, positions


def _delta(
    ref_positions: dict[str, dict[int, int]],
    bt_pos_in: dict[str, dict[int, int]],
    key: str,
    row: dict[str, Any],
) -> int | None:
    """Reference position minus BT score position (positive: BT score ranks it higher)."""
    ref = ref_positions.get(key, {}).get(id(row))
    return None if ref is None else ref - bt_pos_in[key][id(row)]


POSITION_COLUMNS = [
    "benchmark",
    "main_ranking",
    "model",
    "experiment",
    "rank_position",
    "mean_task_position",
    "mean_task_type_position",
    "bt_score_position",
    "bt_score_vs_rank_delta",
    "borda_position",
    "aggregated_position",
    "aggregated_score",
    "bt_score_vs_aggregated_delta",
    "bt_score_vs_borda_delta",
    "agg_borda",
    "agg_bt_score",
    "agg_mean_task",
    "agg_mean_task_type",
    "delta_vs_mean_task",
    "delta_vs_mean_task_type",
    "mean_task",
    "mean_task_type",
    "bt_score",
    "bt_score_low",
    "bt_score_high",
]


def _summary_columns() -> list[tuple[str, str]]:
    """``(stats key, CSV header)`` for the single comparison table."""
    cols = [
        ("benchmark", "Benchmark"),
        ("main_ranking", "Main ranking"),
        ("models", "Models"),
        ("tasks", "Tasks"),
    ]
    for key, label, *_ in REFERENCES:
        cols += [
            (f"{key}_n", f"{label}: models compared"),
            (f"{key}_spearman", f"BT score vs {label}: Spearman"),
            (f"{key}_kendall", f"BT score vs {label}: Kendall"),
            (f"{key}_top_k_overlap", f"BT score vs {label}: top-{TOP_K} overlap"),
            (f"{key}_same_first", f"BT score vs {label}: same top 1"),
            (f"{key}_mover", f"BT score vs {label}: biggest mover"),
        ]
    return cols


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key.endswith(("_spearman", "_kendall")):
        return f"{value:.3f}"
    if key.endswith("_top_k_overlap"):
        return f"{value}/{TOP_K}"
    if key.endswith("_same_first"):
        return "yes" if value else "no"
    return str(value)


def _sorted_stats(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Least rank-agreeing benchmark first."""
    return sorted(stats, key=lambda s: s["rank_spearman"])


def _write_summary_csv(stats: list[dict[str, Any]], path: Path) -> None:
    cols = _summary_columns()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header for _, header in cols])
        for s in _sorted_stats(stats):
            writer.writerow([_fmt(key, s.get(key)) for key, _ in cols])


def _summary_markdown(stats: list[dict[str, Any]]) -> str:
    """Compact markdown view of the table (the CSV also has n and biggest mover)."""
    cols = [
        (key, header)
        for key, header in _summary_columns()
        if not key.endswith(("_mover", "_n"))
    ]
    lines = [
        "| " + " | ".join(header for _, header in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += [
        "| " + " | ".join(_fmt(key, s.get(key)) for key, _ in cols) + " |"
        for s in _sorted_stats(stats)
    ]
    return "\n".join(lines)


def _plot_overview(stats: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(stats, key=lambda s: s["rank_spearman"])
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.22 * len(ordered))))
    ax.barh(
        [s["benchmark"] for s in ordered],
        [s["rank_spearman"] for s in ordered],
        color=["tab:red" if s["rank_spearman"] < 0.9 else "tab:blue" for s in ordered],
    )
    ax.set_xlim(min(0.5, min(s["rank_spearman"] for s in ordered) - 0.05), 1.0)
    ax.set_xlabel("Spearman correlation: rank order vs BT score order")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_benchmark(
    stats: dict[str, Any], positions: list[dict[str, Any]], out: Path, top_n: int
) -> None:
    import matplotlib.pyplot as plt

    rank_pos = [p["rank_position"] for p in positions]
    bt_pos = [p["bt_score_position"] for p in positions]
    n = len(positions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.scatter(rank_pos, bt_pos, s=10, alpha=0.6)
    ax1.plot([1, n], [1, n], color="grey", lw=0.8, ls="--")
    ax1.set_xlabel("Rank position")
    ax1.set_ylabel("BT score position")
    ax1.set_title(
        f"Spearman {stats['rank_spearman']:.3f}, Kendall {stats['rank_kendall']:.3f}"
    )
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    top = sorted(positions, key=lambda p: p["bt_score_position"])[:top_n]
    ys = list(range(len(top)))
    scores = [p["bt_score"] for p in top]
    lo = [
        p["bt_score"] - p["bt_score_low"] if p["bt_score_low"] is not None else 0
        for p in top
    ]
    hi = [
        p["bt_score_high"] - p["bt_score"] if p["bt_score_high"] is not None else 0
        for p in top
    ]
    ax2.errorbar(scores, ys, xerr=[lo, hi], fmt="o", ms=3, capsize=2, lw=1)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(
        [f"{p['model']} (rank {p['rank_position']})" for p in top], fontsize=6
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("BT score (95% bootstrap interval)")
    ax2.set_title(f"Top {len(top)} by BT score")

    fig.suptitle(
        f"{stats['benchmark']}: {stats['models']} models, {stats['tasks']} tasks"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


XLSX_NOTES = (
    "Each benchmark sheet lists every model with a BT score, ordered by the leaderboard "
    "rank, and how its position changes under each ranking system.",
    "",
    "Main ranking: the Rank column header names what the leaderboard's rank is based on "
    "for that benchmark (Borda, or the mean column the benchmark sorts by); the Summary "
    "sheet lists it for every benchmark.",
    "",
    "Columns",
    "  Rank: the leaderboard position (the summary's rank column), by the main ranking.",
    "  Borda position: the leaderboard's own Rank (Borda). On each task a model earns "
    "(number of models - its rank on that task) points; models are ranked by the total.",
    "  Mean (Task) / Mean (TaskType) position: position by that mean, among the models "
    "that have it (models missing a task have no mean). Only present where the "
    "benchmark declares that aggregation.",
    "  BT score position: position by BT score among all models with a BT score.",
    "  Aggregated rank: rank by the average position across Borda, BT score and each "
    "applicable mean (see Input columns). Equivalent to a Borda count over those "
    "rankings. Lower average = better; ties share the best rank.",
    "  Aggregated score (mean position) and Input columns: the average and the four "
    "positions it averages. Inputs are over ALL rated models (not the per-mean subset); "
    "ties share their average position, and a model with no value for a mean takes the "
    "shared tail position of that ranking, after every model that has one.",
    "  Δ vs Rank / Borda / Aggregated = that position - BT score position. Positive: the "
    "BT score ranks the model higher (a better position); negative: lower.",
    "  Δ vs Mean (Task) / Mean (TaskType) = mean position - BT score position, with the "
    "BT score position taken among the same models that have that mean.",
    "  Max |Δ|: the largest absolute change in that row. Sort or filter on it to find "
    "the models that move most.",
    "",
    "BT score: Bradley-Terry score on an Elo-like scale (1000 = average), fitted from "
    "per-task head-to-head wins; a task a model wasn't evaluated on counts as a loss. "
    "Low / high bound a 95% bootstrap interval over tasks. Mean columns are percentages.",
    "Position differences within a bootstrap interval are not meaningful: neighbouring "
    "models often overlap.",
    "Values are computed by scripts/compare_rank_vs_bt_score.py (static, not formulas).",
)


def _sheet_names(names: list[str]) -> dict[str, str]:
    """Excel-safe, unique sheet name (<= 31 chars, no []:*?/\\) per benchmark."""
    used = {"Summary", "Notes"}
    out: dict[str, str] = {}
    for name in names:
        base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
        candidate, i = base, 2
        while candidate.lower() in {u.lower() for u in used}:
            suffix = f" ({i})"
            candidate, i = base[: 31 - len(suffix)] + suffix, i + 1
        used.add(candidate)
        out[name] = candidate
    return out


def _write_xlsx(
    path: Path, stats: list[dict[str, Any]], positions: dict[str, list[dict[str, Any]]]
) -> None:
    """One summary sheet, a notes sheet and one sheet per benchmark."""
    try:
        from openpyxl import Workbook
        from openpyxl.formatting.rule import ColorScaleRule
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.hyperlink import Hyperlink
    except ImportError as e:
        raise SystemExit(
            "--xlsx-out needs openpyxl: run with `uv run --with openpyxl ...`"
        ) from e

    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    header_fill = PatternFill("solid", start_color="DDE3EA")
    sheet_for = _sheet_names([s["benchmark"] for s in stats])

    def style_header(ws, n_cols: int) -> None:
        for c in range(1, n_cols + 1):
            cell = ws.cell(row=1, column=c)
            cell.font, cell.fill = bold, header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    cols = [c for c in _summary_columns() if not c[0].endswith("_mover")]
    ws.append([header for _, header in cols])
    for st in _sorted_stats(stats):
        ws.append([st.get(key) for key, _ in cols])
        link = ws.cell(row=ws.max_row, column=1)
        link.hyperlink = Hyperlink(
            ref=link.coordinate, location=f"'{sheet_for[st['benchmark']]}'!A1"
        )
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = font
            if isinstance(cell.value, float):
                cell.number_format = "0.000"
        row[0].font = Font(name="Arial", size=10, color="0563C1", underline="single")
    style_header(ws, len(cols))
    ws.row_dimensions[1].height = 54
    ws.column_dimensions["A"].width = 30
    for i in range(2, len(cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 14
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    notes = wb.create_sheet("Notes")
    for line in XLSX_NOTES:
        notes.append([line])
    for row in notes.iter_rows():
        row[0].font = bold if row[0].value in ("Columns",) else font
    notes.column_dimensions["A"].width = 130

    for st in _sorted_stats(stats):
        name = st["benchmark"]
        rows = positions[name]
        has_task = st.get("mean_task_n") is not None
        has_type = st.get("mean_task_type_n") is not None
        # (header, row key, number format, width)
        main = st["main_ranking"]
        spec: list[tuple[str, str, str | None, int]] = [
            ("Model", "model", None, 46),
            ("Experiment", "experiment", None, 18),
            (f"Rank (main ranking: {main})", "rank_position", "0", 16),
            ("Borda position", "borda_position", "0", 10),
        ]
        if has_task:
            spec.append(("Mean (Task) position", "mean_task_position", "0", 12))
        if has_type:
            spec.append(
                ("Mean (TaskType) position", "mean_task_type_position", "0", 12)
            )
        spec += [
            ("BT Score position", "bt_score_position", "0", 12),
            ("Aggregated rank", "aggregated_position", "0", 12),
            ("Δ vs Rank", "bt_score_vs_rank_delta", "+0;-0;0", 10),
            ("Δ vs Borda", "bt_score_vs_borda_delta", "+0;-0;0", 10),
        ]
        if has_task:
            spec.append(("Δ vs Mean (Task)", "delta_vs_mean_task", "+0;-0;0", 11))
        if has_type:
            spec.append(
                ("Δ vs Mean (TaskType)", "delta_vs_mean_task_type", "+0;-0;0", 11)
            )
        spec.append(("Δ vs Aggregated", "bt_score_vs_aggregated_delta", "+0;-0;0", 11))
        delta_keys = [k for _, k, f, _ in spec if f == "+0;-0;0"]
        spec += [
            ("Max |Δ|", "max_abs_delta", "0", 9),
            ("BT Score", "bt_score", "0.0", 10),
            ("BT Score low", "bt_score_low", "0.0", 10),
            ("BT Score high", "bt_score_high", "0.0", 10),
        ]
        if has_task:
            spec.append(("Mean (Task)", "mean_task", "0.00%", 11))
        if has_type:
            spec.append(("Mean (TaskType)", "mean_task_type", "0.00%", 11))
        spec.append(("Aggregated score (mean position)", "aggregated_score", "0.0", 14))
        spec.append(("Input: Borda", "agg_borda", "0.0", 10))
        spec.append(("Input: BT Score", "agg_bt_score", "0.0", 10))
        if has_task:
            spec.append(("Input: Mean (Task)", "agg_mean_task", "0.0", 10))
        if has_type:
            spec.append(("Input: Mean (TaskType)", "agg_mean_task_type", "0.0", 11))

        sheet = wb.create_sheet(sheet_for[name])
        sheet.append([h for h, *_ in spec])
        for r in rows:
            deltas = [r[k] for k in delta_keys if r.get(k) is not None]
            r = {**r, "max_abs_delta": max((abs(d) for d in deltas), default=None)}
            sheet.append([r.get(k) for _, k, _, _ in spec])
        for row in sheet.iter_rows(min_row=2):
            for cell, (_, _, fmt, _) in zip(row, spec, strict=True):
                cell.font = font
                if fmt:
                    cell.number_format = fmt
        style_header(sheet, len(spec))
        sheet.row_dimensions[1].height = 42
        for i, (_, _, _, width) in enumerate(spec, 1):
            sheet.column_dimensions[get_column_letter(i)].width = width
        sheet.freeze_panes = "C2"
        sheet.auto_filter.ref = sheet.dimensions
        last = len(rows) + 1
        for i, (_, key, _, _) in enumerate(spec, 1):
            if key in delta_keys:
                col = get_column_letter(i)
                sheet.conditional_formatting.add(
                    f"{col}2:{col}{last}",
                    ColorScaleRule(
                        start_type="num",
                        start_value=-25,
                        start_color="F4A6A6",
                        mid_type="num",
                        mid_value=0,
                        mid_color="FFFFFF",
                        end_type="num",
                        end_value=25,
                        end_color="9FD8A8",
                    ),
                )
    wb.save(path)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        help="Benchmark name; repeat for several. Default: all, including hidden.",
    )
    parser.add_argument("--positions-out", default="rank_vs_bt_score_positions.csv")
    parser.add_argument(
        "--summary-out",
        default="rank_vs_bt_score_summary.csv",
        help="The comparison table as CSV; a .md copy is written next to it.",
    )
    parser.add_argument(
        "--xlsx-out",
        help="Also write an Excel workbook: a summary sheet plus one sheet per "
        "benchmark with every model's position under each ranking system and the "
        "changes (requires openpyxl).",
    )
    parser.add_argument(
        "--plots-dir", help="Write PNG plots here (requires matplotlib)."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_PLOT,
        help="Models shown in the BT score plot.",
    )
    args = parser.parse_args()

    # All benchmarks, including hidden / off-menu ones (as the API preload does).
    names = args.benchmark or [b.name for b in mteb.get_benchmarks()]
    cache = get_cache()

    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict[str, Any]] = []
    all_positions: dict[str, list[dict[str, Any]]] = {}
    with open(args.positions_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSITION_COLUMNS)
        writer.writeheader()
        for name in names:
            try:
                summary = _load_summary(name, cache)
            except Exception as e:  # noqa: BLE001 - keep going across benchmarks
                print(f"skipping {name}: {e}", file=sys.stderr)
                continue
            compared = _compare(name, summary)
            if compared is None:
                continue
            stats, positions = compared
            all_stats.append(stats)
            all_positions[name] = positions
            writer.writerows(positions)
            if plots_dir:
                _plot_benchmark(
                    stats, positions, plots_dir / f"{_slug(name)}.png", args.top_n
                )

    if plots_dir and all_stats:
        _plot_overview(all_stats, plots_dir / "overview.png")
    if args.xlsx_out:
        _write_xlsx(Path(args.xlsx_out), all_stats, all_positions)
    summary_csv = Path(args.summary_out)
    _write_summary_csv(all_stats, summary_csv)
    table = _summary_markdown(all_stats)
    summary_csv.with_suffix(".md").write_text(table + "\n")
    print(table)
    print(
        f"\n{len(all_stats)} benchmarks -> {args.positions_out}, {summary_csv}, "
        f"{summary_csv.with_suffix('.md')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
