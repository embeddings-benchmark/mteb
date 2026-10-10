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

from scipy.stats import kendalltau, spearmanr

import mteb
from mteb.api.aggregators import build_benchmark_summary
from mteb.api.frames import get_cache

if TYPE_CHECKING:
    from mteb.cache.result_cache import ResultCache

TOP_K = 10
MIN_ROWS = 3
TOP_N_PLOT = 30

# (key, label, aggregation that must be declared, row field, higher-is-better).
# ``rank`` is lower-is-better and always available.
REFERENCES: tuple[tuple[str, str, str | None, str, bool], ...] = (
    ("rank", "Rank", None, "rank", False),
    ("mean_task", "Mean (Task)", "mean_task", "mean_task", True),
    ("mean_task_type", "Mean (TaskType)", "mean_task_type", "mean_task_type", True),
)


def _load_summary(name: str, cache: ResultCache) -> dict[str, Any]:
    """Build one benchmark's summary and reduce it to the fields compared here."""
    summary = asyncio.run(build_benchmark_summary(name, cache))
    return {
        "tasks": summary.tasks,
        "aggregations": list(summary.aggregations),
        "rows": [
            {
                "rank": r.rank,
                "model": r.model.name,
                "experiments": r.experiments,
                "mean_task": r.mean_task,
                "mean_task_type": r.mean_task_type,
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


def _compare(
    name: str, summary: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """One summary-table row for the benchmark plus its per-model position rows."""
    rows = [r for r in summary["rows"] if r.get("bt_score") is not None]
    if len(rows) < MIN_ROWS:
        return None

    stats: dict[str, Any] = {
        "benchmark": name,
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
    positions = [
        {
            "benchmark": name,
            "model": r["model"],
            "experiment": json.dumps(r["experiments"]) if r["experiments"] else "",
            "rank_position": ref_positions["rank"][id(r)],
            "mean_task_position": ref_positions.get("mean_task", {}).get(id(r)),
            "mean_task_type_position": ref_positions.get("mean_task_type", {}).get(
                id(r)
            ),
            "bt_score_position": bt_pos[id(r)],
            "bt_score_vs_rank_delta": ref_positions["rank"][id(r)] - bt_pos[id(r)],
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


POSITION_COLUMNS = [
    "benchmark",
    "model",
    "experiment",
    "rank_position",
    "mean_task_position",
    "mean_task_type_position",
    "bt_score_position",
    "bt_score_vs_rank_delta",
    "mean_task",
    "mean_task_type",
    "bt_score",
    "bt_score_low",
    "bt_score_high",
]


def _summary_columns() -> list[tuple[str, str]]:
    """``(stats key, CSV header)`` for the single comparison table."""
    cols = [("benchmark", "Benchmark"), ("models", "Models"), ("tasks", "Tasks")]
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
            writer.writerows(positions)
            if plots_dir:
                _plot_benchmark(
                    stats, positions, plots_dir / f"{_slug(name)}.png", args.top_n
                )

    if plots_dir and all_stats:
        _plot_overview(all_stats, plots_dir / "overview.png")
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
