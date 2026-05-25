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

def task_category(task_name: str, mteb_task_type: str) -> str:
    """Map mteb task type → display category."""
    return {
        "Any2AnyRetrieval": "Retr",
        "VideoCentricQA": "QA",
        "VideoClassification": "Cls",
        "VideoClustering": "Clust",
        "VideoPairClassification": "Pair",
        "VideoZeroshotClassification": "ZS",
    }.get(mteb_task_type, mteb_task_type)


CATEGORY_ORDER = ["Retr", "QA", "Cls", "Clust", "Pair", "ZS"]


# ---------------------------------------------------------------------------
# Which scope each model belongs to (its natural evaluation surface).
# Models with audio + video + text capability go on MVEB. Text + video
# models go on MVEB(text-video). Video-only encoders go on MVEB(video).
# Each scope's leaderboard table lists only the models that naturally
# belong there — no cross-scope comparison.
# ---------------------------------------------------------------------------

MODEL_SCOPE: dict[str, str] = {
    # Audio + video + text (MVEB):
    "e5-omni-3B": "mveb",
    "e5-omni-7B": "mveb",
    "LCO-Embedding-Omni-3B": "mveb",
    "LCO-Embedding-Omni-7B": "mveb",
    "Qwen2.5-Omni-3B": "mveb",
    "Qwen2.5-Omni-7B": "mveb",
    "OmniEmbed-v0.1": "mveb",
    "omni-embed-nemotron-3b": "mveb",
    "jina-embeddings-v5-omni-nano": "mveb",
    "jina-embeddings-v5-omni-small": "mveb",
    "ebind-audio-vision": "mveb",
    "ebind-full": "mveb",
    "pe-av-small": "mveb",
    "pe-av-base": "mveb",
    "pe-av-large": "mveb",
    # Text + video (MVEB(text-video)):
    "ebind-points-vision": "tv",
    "UME-R1-2B": "tv",
    "UME-R1-7B": "tv",
    "xclip-base-patch16": "tv",
    "xclip-base-patch32": "tv",
    "xclip-large-patch14": "tv",
    # Video-only (MVEB(video)):
    "vjepa2-vitg-fpc64-256": "v",
    "vjepa2-vitg-fpc64-384": "v",
    "vjepa2-vitg-fpc64-384-ssv2": "v",
    "vjepa2-vitg-fpc32-384-diving48": "v",
    "vjepa2-vith-fpc64-256": "v",
    "vjepa2-vitl-fpc64-256": "v",
    "vjepa2-vitl-fpc16-256-ssv2": "v",
    "vjepa2-vitl-fpc32-256-diving48": "v",
}


# Embedding paradigm shown in the leaderboard "Type" column.
# Labels match the family table in section 2 (Methodology).
MODEL_FAMILY: dict[str, str] = {
    "e5-omni-3B": "MLLM Embed",
    "e5-omni-7B": "MLLM Embed",
    "LCO-Embedding-Omni-3B": "MLLM Embed",
    "LCO-Embedding-Omni-7B": "MLLM Embed",
    "OmniEmbed-v0.1": "MLLM Embed",
    "omni-embed-nemotron-3b": "MLLM Embed",
    "jina-embeddings-v5-omni-nano": "MLLM Embed",
    "jina-embeddings-v5-omni-small": "MLLM Embed",
    "UME-R1-2B": "MLLM Embed",
    "UME-R1-7B": "MLLM Embed",
    "Qwen2.5-Omni-3B": "Gen-MLLM",
    "Qwen2.5-Omni-7B": "Gen-MLLM",
    "ebind-audio-vision": "Multi-Bind",
    "ebind-full": "Multi-Bind",
    "ebind-points-vision": "Multi-Bind",
    "pe-av-small": "Aud-Vis Contr",
    "pe-av-base": "Aud-Vis Contr",
    "pe-av-large": "Aud-Vis Contr",
    "xclip-base-patch16": "Vid-Text Contr",
    "xclip-base-patch32": "Vid-Text Contr",
    "xclip-large-patch14": "Vid-Text Contr",
    "vjepa2-vitg-fpc64-256": "Self-sup Vid",
    "vjepa2-vitg-fpc64-384": "Self-sup Vid",
    "vjepa2-vitg-fpc64-384-ssv2": "Self-sup Vid",
    "vjepa2-vitg-fpc32-384-diving48": "Self-sup Vid",
    "vjepa2-vith-fpc64-256": "Self-sup Vid",
    "vjepa2-vitl-fpc64-256": "Self-sup Vid",
    "vjepa2-vitl-fpc16-256-ssv2": "Self-sup Vid",
    "vjepa2-vitl-fpc32-256-diving48": "Self-sup Vid",
}


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

def _fmt_params(n_params: int | None) -> str:
    """Format param count as "847M" or "8.9B"."""
    if n_params is None or pd.isna(n_params):
        return "--"
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.1f}B"
    return f"{n_params / 1_000_000:.0f}M"


def _model_params(model_name: str) -> int | None:
    """Look up parameter count via mteb model metadata; None if unknown."""
    try:
        meta = mteb.get_model_meta(model_name)
        return meta.n_parameters
    except Exception:
        return None


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


SCOPE_META = {
    "mveb":    {"name": "MVEB",              "label": "tab:mveb-main-results",        "caption_scope": "MVEB"},
    "ext":     {"name": "MVEB(extended)",    "label": "tab:mveb-extended-results",    "caption_scope": "MVEB(extended)"},
    "tv":      {"name": "MVEB(text, video)", "label": "tab:mveb-text-video-results", "caption_scope": "MVEB(text, video)"},
    "v":       {"name": "MVEB(video)",       "label": "tab:mveb-video-results",       "caption_scope": "MVEB(video)"},
}


def emit_scope_table(
    df: pd.DataFrame,
    scope_tasks: list[str],
    scope_key: str,
    scope_models: list[str],
    task_type_lookup: dict[str, str],
    output_path: Path,
) -> None:
    """Render a single leaderboard table for one scope.

    Only models in `scope_models` (those that naturally belong on this scope)
    are included. Borda rank is computed within those models on the scope's
    tasks.
    """
    # Restrict df to this scope's models
    in_scope = [m for m in scope_models if m in df.index]
    if not in_scope:
        print(f"  [{scope_key}] no models with results — skipping")
        return
    df = df.loc[in_scope].copy()

    available = [t for t in scope_tasks if t in df.columns]
    _, rank = borda_ranks(df, scope_tasks)

    # Per-category averages over the scope's tasks
    cat_avgs = category_averages(df, scope_tasks, task_type_lookup)
    df["mean"] = df[available].mean(axis=1) * 100
    df["macro"] = pd.DataFrame({c: s for c, s in cat_avgs.items()}).mean(axis=1)

    # Sort by Borda rank
    df = df.assign(_r=rank).sort_values("_r", na_position="last").drop(columns="_r")
    rank = rank.reindex(df.index)

    # Bests (global within this table)
    best = {
        "mean": df["mean"].max(),
        "macro": df["macro"].max(),
        "rank": rank.min(),
    }
    for cat in CATEGORY_ORDER:
        if cat in cat_avgs:
            best[cat] = cat_avgs[cat].max()

    cats_present = [c for c in CATEGORY_ORDER if c in cat_avgs]
    cat_counts: dict[str, int] = defaultdict(int)
    for t in available:
        if t in task_type_lookup:
            cat_counts[task_type_lookup[t]] += 1

    meta = SCOPE_META[scope_key]
    cap_scope = meta["caption_scope"]
    n_tasks = len(available)
    n_models = len(df)

    cat_glossary = (
        ", ".join({
            "Retr": r"Retr (retrieval; per-direction breakdown in appendix)",
            "QA": "QA",
            "Cls": "Cls (classification)",
            "Clust": "Clust (clustering)",
            "Pair": "Pair (pair classification)",
            "ZS": "ZS (zero-shot classification)",
        }[c] for c in cats_present)
    )

    out: list[str] = []
    out.append(r"% AUTOGENERATED by scripts/mveb_paper/generate_main_results_table.py — do not edit by hand.")
    out.append(r"\begin{table*}[!th]")
    out.append(r"    \centering")
    out.append(r"    \resizebox{\textwidth}{!}{\setlength{\tabcolsep}{4pt}{\footnotesize")
    col_spec = "l" + "ll" + "|" + "c" + "|" + "c"*2 + "|" + "c"*len(cats_present)
    ncols = 1 + 2 + 1 + 2 + len(cats_present)
    out.append(r"    \begin{tabular}{" + col_spec + r"}")
    out.append(r"    \toprule")
    out.append(
        r"     & & & \textbf{Rank} ($\downarrow$) & \multicolumn{2}{c|}{\textbf{Average}} & "
        r"\multicolumn{" + str(len(cats_present)) + r"}{c}{\textbf{Average per Category}} \\"
    )
    out.append(r"    \cmidrule(r){4-4} \cmidrule(lr){5-6} \cmidrule(l){7-" + str(ncols) + r"}")
    cat_headers = " & ".join(f"\\textbf{{{c}}}" for c in cats_present)
    out.append(r"    \textbf{Model} & \textbf{Type} & \textbf{Params} & " + cap_scope + r" & Mean & Macro & " + cat_headers + r" \\")
    out.append(r"    \midrule")
    count_cells = " & ".join(f"\\textcolor{{gray}}{{({cat_counts[c]})}}" for c in cats_present)
    out.append(
        r"    \textcolor{gray}{Number of tasks} & & & & & & " + count_cells + r" \\"
    )
    out.append(r"    \midrule")

    for m in df.index:
        row = df.loc[m]
        short = m.split("/")[-1]
        display = short.replace("_", "\\_")
        family = MODEL_FAMILY.get(short, "--")
        params = _fmt_params(_model_params(m))
        cells = [
            display,
            family,
            params,
            _fmt_rank(rank.loc[m], best["rank"], None),
            _fmt_score(row["mean"], best["mean"], None),
            _fmt_score(row["macro"], best["macro"], None),
        ]
        for cat in cats_present:
            v = cat_avgs[cat].get(m, np.nan)
            cells.append(_fmt_score(v, best.get(cat), None))
        out.append("    " + " & ".join(cells) + r" \\")

    out.append(r"    \bottomrule")
    out.append(r"    \end{tabular}}}")
    out.append(
        r"    \caption{Models on \textbf{" + cap_scope + r"} ranked by Borda count over "
        f"{n_tasks} tasks. "
        r"\textbf{Mean} is the arithmetic mean over the model's evaluated tasks; "
        r"\textbf{Macro} is the macro-average across task categories (each category "
        r"weighted equally regardless of task count). Task categories: " + cat_glossary +
        r". Model types are defined in Table~\ref{tab:model-families}. "
        r"\textbf{Bold} = best per column.}"
    )
    out.append(r"    \label{" + meta["label"] + r"}")
    out.append(r"\end{table*}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out) + "\n")
    print(f"  [{scope_key}] {n_models} models, {n_tasks} tasks → {output_path.name}")


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
        "--tables-dir", type=Path,
        default=Path("/Users/adnan/research/mveb/699f18a1b6536d2f9f06230a/tables"),
        help="Where to write the LaTeX tables.",
    )
    args = parser.parse_args()

    mveb = mteb.get_benchmark("MVEB")
    mveb_ext = mteb.get_benchmark("MVEB(extended)")
    mveb_tv = mteb.get_benchmark("MVEB(text, video)")
    mveb_v = mteb.get_benchmark("MVEB(video)")

    tasks_by_scope = {
        "mveb": [t.metadata.name for t in mveb.tasks],
        "ext":  [t.metadata.name for t in mveb_ext.tasks],
        "tv":   [t.metadata.name for t in mveb_tv.tasks],
        "v":    [t.metadata.name for t in mveb_v.tasks],
    }

    task_type_lookup: dict[str, str] = {}
    for b in (mveb, mveb_ext, mveb_tv, mveb_v):
        for t in b.tasks:
            task_type_lookup[t.metadata.name] = task_category(
                t.metadata.name, t.metadata.type
            )

    all_tasks: set[str] = set()
    for ts in tasks_by_scope.values():
        all_tasks.update(ts)

    print(f"Loading results for {len(all_tasks)} unique tasks from {args.results_dir}")
    df = load_results(args.results_dir, all_tasks)
    print(f"Loaded {len(df)} models with at least one task result")

    # Group models by their natural scope. MVEB(extended) is the audio+video+text
    # superset, so it shares the mveb-scope model list.
    models_by_scope: dict[str, list[str]] = {"mveb": [], "tv": [], "v": []}
    for m in df.index:
        short = m.split("/")[-1]
        scope = MODEL_SCOPE.get(short)
        if scope in models_by_scope:
            models_by_scope[scope].append(m)
        else:
            print(f"  unknown model (skipped): {m}")
    models_by_scope["ext"] = list(models_by_scope["mveb"])

    outputs = {
        "mveb": args.tables_dir / "mveb_main_results.tex",
        "ext":  args.tables_dir / "mveb_extended_results.tex",
        "tv":   args.tables_dir / "mveb_text_video_results.tex",
        "v":    args.tables_dir / "mveb_video_results.tex",
    }
    for scope_key in ("mveb", "ext", "tv", "v"):
        emit_scope_table(
            df,
            tasks_by_scope[scope_key],
            scope_key,
            models_by_scope[scope_key],
            task_type_lookup,
            outputs[scope_key],
        )


if __name__ == "__main__":
    main()
