#!/usr/bin/env python3
"""Generate the main MVEB results table (LaTeX).

Reads results directly from JSON files in `--results-dir` to avoid cache
filtering issues. Writes the leaderboard table for MVEB (the audio-video
master benchmark) with MVEB(text-video) and MVEB(video) ranks as auxiliary
columns, grouped by model family.

Output: tables/mveb_main_results.tex (in the paper directory).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("mteb").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import mteb


# ---------------------------------------------------------------------------
# Task type → display category
# ---------------------------------------------------------------------------

# Retrieval split into audio-conditioned vs text-video, because the AV
# directions are MVEB's differentiator.
_AUDIO_RETRIEVAL_SUFFIXES = (
    "A2VRetrieval", "V2ARetrieval", "T2VARetrieval", "VA2TRetrieval",
    "AT2VRetrieval", "VT2ARetrieval",
    "A2V", "V2A", "T2VA", "VA2T", "AT2V", "VT2A",  # MSRVTT naming
)


def task_category(task_name: str, mteb_task_type: str) -> str:
    """Map mteb task type → display category. Retrieval is split into
    'AV-Retrieval' and 'TV-Retrieval' for richer reporting."""
    if mteb_task_type == "Any2AnyRetrieval":
        return "AV-Retr" if any(task_name.endswith(s) for s in _AUDIO_RETRIEVAL_SUFFIXES) else "TV-Retr"
    return {
        "VideoCentricQA": "QA",
        "VideoClassification": "Cls",
        "VideoClustering": "Clust",
        "VideoPairClassification": "Pair",
        "VideoZeroshotClassification": "ZS",
    }.get(mteb_task_type, mteb_task_type)


CATEGORY_ORDER = ["TV-Retr", "AV-Retr", "QA", "Cls", "Clust", "Pair", "ZS"]


# ---------------------------------------------------------------------------
# Model family grouping for the leaderboard
# ---------------------------------------------------------------------------

# Display name (stripped of org/) → group
MODEL_GROUP: dict[str, str] = {
    # Omni multimodal (a, i, t, v)
    "e5-omni-3B": "Omni multimodal",
    "e5-omni-7B": "Omni multimodal",
    "LCO-Embedding-Omni-3B": "Omni multimodal",
    "LCO-Embedding-Omni-7B": "Omni multimodal",
    "Qwen2.5-Omni-3B": "Omni multimodal",
    "Qwen2.5-Omni-7B": "Omni multimodal",
    "OmniEmbed-v0.1": "Omni multimodal",
    "omni-embed-nemotron-3b": "Omni multimodal",
    "jina-embeddings-v5-omni-nano": "Omni multimodal",
    "jina-embeddings-v5-omni-small": "Omni multimodal",
    # Audio-visual specialized
    "ebind-audio-vision": "Audio-visual",
    "ebind-full": "Audio-visual",
    "pe-av-small": "Audio-visual",
    "pe-av-base": "Audio-visual",
    "pe-av-large": "Audio-visual",
    # Vision-language (no audio)
    "ebind-points-vision": "Vision-language",
    "UME-R1-2B": "Vision-language",
    "UME-R1-7B": "Vision-language",
    "xclip-base-patch16": "Vision-language",
    "xclip-base-patch32": "Vision-language",
    "xclip-large-patch14": "Vision-language",
    # Vision-only (no audio, no text encoder)
    "vjepa2-vitg-fpc64-256": "Vision-only",
    "vjepa2-vitg-fpc64-384": "Vision-only",
    "vjepa2-vitg-fpc64-384-ssv2": "Vision-only",
    "vjepa2-vitg-fpc32-384-diving48": "Vision-only",
    "vjepa2-vith-fpc64-256": "Vision-only",
    "vjepa2-vitl-fpc64-256": "Vision-only",
    "vjepa2-vitl-fpc16-256-ssv2": "Vision-only",
    "vjepa2-vitl-fpc32-256-diving48": "Vision-only",
}

MODEL_GROUP_ORDER = ["Omni multimodal", "Audio-visual", "Vision-language", "Vision-only", "Other"]


# ---------------------------------------------------------------------------
# Result loading (fixes the MAEB single-subset bug)
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, task_names: set[str]) -> pd.DataFrame:
    """Read each model's task JSON files and build a model-x-task score matrix.

    For tasks with multiple subsets on the test split, averages across subsets
    (e.g. VATEX's en/zh subsets both contribute).
    """
    rows: dict[str, dict] = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        revs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not revs:
            continue
        rev_dir = revs[0]
        row: dict = {"model": model_name}
        for task_file in rev_dir.glob("*.json"):
            task_name = task_file.stem
            if task_name not in task_names:
                continue
            try:
                data = json.loads(task_file.read_text())
            except json.JSONDecodeError:
                continue
            scores = data.get("scores", {})
            # Prefer test, fall back to val/train, take whichever split exists
            split_data = None
            for split in ("test", "val", "validation", "dev", "train"):
                if scores.get(split):
                    split_data = scores[split]
                    break
            if not split_data or not isinstance(split_data, list):
                continue
            subset_scores = [
                s.get("main_score") for s in split_data
                if isinstance(s, dict) and s.get("main_score") is not None
            ]
            if subset_scores:
                row[task_name] = float(np.mean(subset_scores))
        if len(row) > 1:
            rows[model_name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Borda count
# ---------------------------------------------------------------------------

def borda_ranks(df: pd.DataFrame, task_names: list[str]) -> tuple[pd.Series, pd.Series]:
    """Per-model Borda count over `task_names`. Returns (borda_score, rank)."""
    available = [t for t in task_names if t in df.columns]
    borda = pd.Series(0.0, index=df.index)
    for t in available:
        s = df[t].dropna()
        if len(s) < 2:
            continue
        ranks = s.rank(ascending=False, method="min")
        borda[s.index] += len(s) - ranks + 1
    rank = borda.rank(ascending=False, method="min").astype("Int64")
    return borda, rank


# ---------------------------------------------------------------------------
# Per-category averages
# ---------------------------------------------------------------------------

def category_averages(
    df: pd.DataFrame, task_names: list[str], task_type_lookup: dict[str, str]
) -> dict[str, pd.Series]:
    """{category: per-model mean score over tasks in that category} (×100)."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for t in task_names:
        if t in df.columns and t in task_type_lookup:
            by_cat[task_type_lookup[t]].append(t)
    return {
        cat: df[tasks].mean(axis=1) * 100
        for cat, tasks in by_cat.items()
    }


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def _fmt_score(value: float, global_best: float, group_best: float) -> str:
    if pd.isna(value):
        return "-"
    cell = ""
    if group_best is not None and not pd.isna(group_best) and abs(value - group_best) < 0.05:
        cell = "\\cellcolor{gray!20}"
    if global_best is not None and not pd.isna(global_best) and abs(value - global_best) < 0.05:
        return f"{cell}\\textbf{{{value:.1f}}}"
    return f"{cell}{value:.1f}"


def _fmt_rank(rank, global_best, group_best) -> str:
    if pd.isna(rank):
        return "-"
    cell = ""
    if group_best is not None and rank == group_best:
        cell = "\\cellcolor{gray!20}"
    if global_best is not None and rank == global_best:
        return f"{cell}\\textbf{{{int(rank)}}}"
    return f"{cell}{int(rank)}"


def emit_table(
    df: pd.DataFrame,
    mveb_tasks: list[str],
    tv_tasks: list[str],
    v_tasks: list[str],
    task_type_lookup: dict[str, str],
    output_path: Path,
) -> None:
    # Ranks per benchmark
    _, mveb_rank = borda_ranks(df, mveb_tasks)
    _, tv_rank = borda_ranks(df, tv_tasks)
    _, v_rank = borda_ranks(df, v_tasks)

    # Filter to models with at least one MVEB result
    available_mveb = [t for t in mveb_tasks if t in df.columns]
    keep = df[available_mveb].notna().any(axis=1)
    df = df[keep].copy()
    mveb_rank = mveb_rank.reindex(df.index)
    tv_rank = tv_rank.reindex(df.index)
    v_rank = v_rank.reindex(df.index)

    # Category averages over MVEB tasks
    cat_avgs = category_averages(df, mveb_tasks, task_type_lookup)
    # Overall mean (over MVEB tasks)
    df["mean"] = df[available_mveb].mean(axis=1) * 100
    # Weighted (macro across categories)
    cat_stack = pd.DataFrame({c: s for c, s in cat_avgs.items()})
    df["weighted_mean"] = cat_stack.mean(axis=1)

    # Sort by MVEB rank (low number = better)
    df = df.assign(_rank=mveb_rank).sort_values("_rank", na_position="last").drop(columns="_rank")
    mveb_rank = mveb_rank.reindex(df.index)
    tv_rank = tv_rank.reindex(df.index)
    v_rank = v_rank.reindex(df.index)

    # Group models
    def model_group(model: str) -> str:
        short = model.split("/")[-1]
        return MODEL_GROUP.get(short, "Other")

    df["group"] = df["model"].apply(model_group)
    groups: dict[str, pd.Index] = {
        g: df.index[df["group"] == g]
        for g in MODEL_GROUP_ORDER
        if (df["group"] == g).any()
    }

    # Global bests
    g_best = {
        "mean": df["mean"].max(),
        "weighted_mean": df["weighted_mean"].max(),
        "mveb_rank": mveb_rank.min(),
        "tv_rank": tv_rank.min(),
        "v_rank": v_rank.min(),
    }
    for cat in CATEGORY_ORDER:
        if cat in cat_avgs:
            g_best[cat] = cat_avgs[cat].max()

    # Per-group bests
    group_best: dict[str, dict] = {}
    for g, idx in groups.items():
        gb = {
            "mean": df.loc[idx, "mean"].max(),
            "weighted_mean": df.loc[idx, "weighted_mean"].max(),
            "mveb_rank": mveb_rank.loc[idx].min(),
            "tv_rank": tv_rank.loc[idx].min(),
            "v_rank": v_rank.loc[idx].min(),
        }
        for cat in CATEGORY_ORDER:
            if cat in cat_avgs:
                gb[cat] = cat_avgs[cat].loc[idx].max()
        group_best[g] = gb

    cats_present = [c for c in CATEGORY_ORDER if c in cat_avgs]
    cat_counts: dict[str, int] = defaultdict(int)
    for t in available_mveb:
        if t in task_type_lookup:
            cat_counts[task_type_lookup[t]] += 1

    # Build LaTeX
    out: list[str] = []
    out.append(r"% AUTOGENERATED by scripts/mveb_paper/generate_main_results_table.py — do not edit by hand.")
    out.append(r"\begin{table*}[!th]")
    out.append(r"    \centering")
    out.append(
        r"    \caption{"
        r"Models on MVEB (audio-video, " + str(len(available_mveb)) + r" tasks) ranked by Borda count. "
        r"The \textbf{TV} and \textbf{V} rank columns show each model's rank on MVEB(text-video) and MVEB(video) respectively. "
        r"\textbf{All} is the arithmetic mean over MVEB tasks; \textbf{Cat.} is the macro-average across task categories. "
        r"Task categories: TV-Retr (text-video retrieval), AV-Retr (audio-conditioned retrieval), QA, Cls (classification), "
        r"Clust (clustering), Pair (pair classification), ZS (zero-shot classification). "
        r"Best score per column in \textbf{bold}; best within model group highlighted in grey.}"
    )
    out.append(r"    \label{tab:mveb-main-results}")
    out.append(r"    \resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}{\footnotesize")
    ncols = 1 + 3 + 2 + len(cats_present)
    col_spec = "l" + "c"*3 + "|" + "c"*2 + "|" + "c"*len(cats_present)
    out.append(r"    \begin{tabular}{" + col_spec + r"}")
    out.append(r"    \toprule")
    out.append(
        r"     & \multicolumn{3}{c|}{\textbf{Rank} ($\downarrow$)} & "
        r"\multicolumn{2}{c|}{\textbf{Average}} & "
        r"\multicolumn{" + str(len(cats_present)) + r"}{c}{\textbf{Average per Category}} \\"
    )
    out.append(r"    \cmidrule(r){2-4} \cmidrule(lr){5-6} \cmidrule(l){7-" + str(ncols) + r"}")
    cat_headers = " & ".join(f"\\textbf{{{c}}}" for c in cats_present)
    out.append(r"    \textbf{Model} & MVEB & TV & V & All & Cat. & " + cat_headers + r" \\")
    out.append(r"    \midrule")
    # Count row
    count_cells = " & ".join(f"\\textcolor{{gray}}{{({cat_counts[c]})}}" for c in cats_present)
    out.append(
        r"    \textcolor{gray}{Number of tasks} & & & & "
        f"\\textcolor{{gray}}{{({len(available_mveb)})}} & "
        f"\\textcolor{{gray}}{{({len(available_mveb)})}} & {count_cells} \\\\"
    )
    out.append(r"    \midrule")

    for g_i, group in enumerate(MODEL_GROUP_ORDER):
        if group not in groups:
            continue
        idx = groups[group]
        out.append(r"    \textbf{" + group + r"} \\")
        out.append(r"    \midrule")
        # Sort within group by MVEB rank
        idx_sorted = mveb_rank.loc[idx].sort_values(na_position="last").index
        gb = group_best[group]
        for m in idx_sorted:
            row = df.loc[m]
            display = m.split("/")[-1]
            display = display.replace("_", "\\_")
            cells = [
                display,
                _fmt_rank(mveb_rank.loc[m], g_best["mveb_rank"], gb.get("mveb_rank")),
                _fmt_rank(tv_rank.loc[m], g_best["tv_rank"], gb.get("tv_rank")),
                _fmt_rank(v_rank.loc[m], g_best["v_rank"], gb.get("v_rank")),
                _fmt_score(row["mean"], g_best["mean"], gb.get("mean")),
                _fmt_score(row["weighted_mean"], g_best["weighted_mean"], gb.get("weighted_mean")),
            ]
            for cat in cats_present:
                v = cat_avgs[cat].get(m, np.nan)
                cells.append(_fmt_score(v, g_best.get(cat), gb.get(cat)))
            out.append("    " + " & ".join(cells) + r" \\")
        # Midrule between groups (except last)
        remaining = [
            g for g in MODEL_GROUP_ORDER[g_i+1:]
            if g in groups
        ]
        if remaining:
            out.append(r"    \midrule")

    out.append(r"    \bottomrule")
    out.append(r"    \end{tabular}}}")
    out.append(r"\end{table*}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/Users/adnan/research/mveb/results/results"),
        help="Per-model results directory (e.g. embeddings-benchmark/results clone).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/Users/adnan/research/mveb/699f18a1b6536d2f9f06230a/tables/mveb_main_results.tex"),
        help="Where to write the LaTeX table.",
    )
    args = parser.parse_args()

    mveb = mteb.get_benchmark("MVEB")
    mveb_tv = mteb.get_benchmark("MVEB(text, video)")
    mveb_v = mteb.get_benchmark("MVEB(video)")

    mveb_tasks = [t.metadata.name for t in mveb.tasks]
    tv_tasks = [t.metadata.name for t in mveb_tv.tasks]
    v_tasks = [t.metadata.name for t in mveb_v.tasks]

    # Task type lookup across all benchmark tasks
    task_type_lookup: dict[str, str] = {}
    for b in (mveb, mveb_tv, mveb_v):
        for t in b.tasks:
            task_type_lookup[t.metadata.name] = task_category(
                t.metadata.name, t.metadata.type
            )

    all_tasks = set(mveb_tasks) | set(tv_tasks) | set(v_tasks)
    print(f"Loading results for {len(all_tasks)} unique tasks from {args.results_dir}")
    df = load_results(args.results_dir, all_tasks)
    print(f"Loaded {len(df)} models with at least one task result")

    emit_table(df, mveb_tasks, tv_tasks, v_tasks, task_type_lookup, args.output)
    print(f"Wrote table to {args.output}")


if __name__ == "__main__":
    main()
