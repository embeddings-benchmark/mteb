"""Compare leaderboard rank order with ELO order for every benchmark.

Builds each benchmark's summary in-process with the same code the API uses
(``mteb.api.aggregators.build_benchmark_summary`` over the local ``ResultCache`` /
cached results frames), so no running server is needed:

    python scripts/compare_rank_vs_elo.py
    python scripts/compare_rank_vs_elo.py --benchmark "MTEB(eng, v2)" --benchmark LongEmbed

Pass ``--plots-dir DIR`` (needs matplotlib, in the ``leaderboard`` extra) to also
write ``overview.png`` (Spearman per benchmark) and one ``<benchmark>.png`` each with
a rank-vs-ELO position scatter and the top models' ELO with 95% bootstrap intervals.

Writes two files:
  * ``--positions-out`` (default ``rank_vs_elo_positions.csv``): one row per
    (benchmark, model, experiment) with its rank position, ELO position, the
    difference, and the underlying scores.
  * ``--summary-out`` (default ``rank_vs_elo_summary.md``): one row per benchmark
    with Spearman / Kendall correlation, top-10 overlap, whether #1 agrees and the
    biggest mover. The same table is printed to stdout.

``rank position`` is the order of the summary's ``rank`` field; ``ELO position`` is the
order by ``elo`` (descending). Both are 1-based within a benchmark. Rows without an
``elo`` are skipped, as are benchmarks with fewer than three rated rows.
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
TOP_N_ELO = 30


def _load_summary(name: str, cache: ResultCache) -> dict[str, Any]:
    """Build one benchmark's summary and reduce it to the fields compared here."""
    summary = asyncio.run(build_benchmark_summary(name, cache))
    return {
        "tasks": summary.tasks,
        "rows": [
            {
                "rank": r.rank,
                "model": {"name": r.model.name},
                "experiments": r.experiments,
                "meanTask": r.mean_task,
                "elo": r.elo,
                "eloLow": r.elo_low,
                "eloHigh": r.elo_high,
            }
            for r in summary.rows
        ],
    }


def _compare(
    name: str, summary: dict[str, Any]
) -> tuple[dict[str, Any], list[list]] | None:
    rows = [r for r in summary["rows"] if r.get("elo") is not None]
    if len(rows) < MIN_ROWS:
        return None

    by_rank = sorted(rows, key=lambda r: r["rank"])
    rank_pos = {id(r): i + 1 for i, r in enumerate(by_rank)}
    elo_pos = {
        id(r): i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["elo"]))
    }

    rp = [rank_pos[id(r)] for r in rows]
    ep = [elo_pos[id(r)] for r in rows]
    top_elo = {id(r) for r in rows if elo_pos[id(r)] <= TOP_K}
    top_rank = [r for r in rows if rank_pos[id(r)] <= TOP_K]
    mover = max(rows, key=lambda r: abs(rank_pos[id(r)] - elo_pos[id(r)]))

    stats = {
        "benchmark": name,
        "models": len(rows),
        "tasks": len(summary["tasks"]),
        "spearman": float(spearmanr(rp, ep)[0]),
        "kendall": float(kendalltau(rp, ep)[0]),
        "top_k_overlap": sum(id(r) in top_elo for r in top_rank),
        "same_first": by_rank[0] is min(rows, key=lambda r: elo_pos[id(r)]),
        "mover": (
            f"{mover['model']['name']} "
            f"(rank {rank_pos[id(mover)]} → ELO {elo_pos[id(mover)]})"
        ),
    }
    positions = [
        [
            name,
            r["model"]["name"],
            json.dumps(r["experiments"]) if r.get("experiments") else "",
            rank_pos[id(r)],
            elo_pos[id(r)],
            rank_pos[id(r)] - elo_pos[id(r)],
            r.get("meanTask"),
            round(r["elo"], 1),
            None if r.get("eloLow") is None else round(r["eloLow"], 1),
            None if r.get("eloHigh") is None else round(r["eloHigh"], 1),
        ]
        for r in by_rank
    ]
    return stats, positions


def _summary_table(stats: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Models | Tasks | Spearman | Kendall | "
        f"Top-{TOP_K} overlap | Same #1 | Biggest mover |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(stats, key=lambda s: s["spearman"]):
        lines.append(
            f"| {s['benchmark']} | {s['models']} | {s['tasks']} | "
            f"{s['spearman']:.3f} | {s['kendall']:.3f} | "
            f"{s['top_k_overlap']}/{TOP_K} | {'yes' if s['same_first'] else 'no'} | "
            f"{s['mover']} |"
        )
    return "\n".join(lines)


def _plot_overview(stats: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(stats, key=lambda s: s["spearman"])
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.22 * len(ordered))))
    ax.barh(
        [s["benchmark"] for s in ordered],
        [s["spearman"] for s in ordered],
        color=["tab:red" if s["spearman"] < 0.9 else "tab:blue" for s in ordered],
    )
    ax.set_xlim(min(0.5, min(s["spearman"] for s in ordered) - 0.05), 1.0)
    ax.set_xlabel("Spearman correlation: rank order vs ELO order")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_benchmark(
    stats: dict[str, Any], positions: list[list], out: Path, top_n: int
) -> None:
    import matplotlib.pyplot as plt

    rank_pos = [p[3] for p in positions]
    elo_pos = [p[4] for p in positions]
    n = len(positions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.scatter(rank_pos, elo_pos, s=10, alpha=0.6)
    ax1.plot([1, n], [1, n], color="grey", lw=0.8, ls="--")
    ax1.set_xlabel("Rank position")
    ax1.set_ylabel("ELO position")
    ax1.set_title(f"Spearman {stats['spearman']:.3f}, Kendall {stats['kendall']:.3f}")
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    top = sorted(positions, key=lambda p: p[4])[:top_n]
    ys = list(range(len(top)))
    elo = [p[7] for p in top]
    lo = [p[7] - p[8] if p[8] is not None else 0 for p in top]
    hi = [p[9] - p[7] if p[9] is not None else 0 for p in top]
    ax2.errorbar(elo, ys, xerr=[lo, hi], fmt="o", ms=3, capsize=2, lw=1)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{p[1]} (rank {p[3]})" for p in top], fontsize=6)
    ax2.invert_yaxis()
    ax2.set_xlabel("ELO (95% bootstrap interval)")
    ax2.set_title(f"Top {len(top)} by ELO")

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
    parser.add_argument("--positions-out", default="rank_vs_elo_positions.csv")
    parser.add_argument("--summary-out", default="rank_vs_elo_summary.md")
    parser.add_argument(
        "--plots-dir", help="Write PNG plots here (requires matplotlib)."
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_ELO, help="Models shown in the ELO plot."
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
        writer = csv.writer(f)
        writer.writerow(
            [
                "benchmark",
                "model",
                "experiment",
                "rank_position",
                "elo_position",
                "delta",
                "mean_task",
                "elo",
                "elo_low",
                "elo_high",
            ]
        )
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
    table = _summary_table(all_stats)
    with open(args.summary_out, "w") as f:
        f.write(table + "\n")
    print(table)
    print(
        f"\n{len(all_stats)} benchmarks -> {args.positions_out}, {args.summary_out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
