#!/usr/bin/env python3
"""Per-task appendix tables (retrieval-by-direction + per-family).

Loads results through ``mteb.ResultCache`` (defaults to ``~/.cache/mteb`` and
pulls the remote ``embeddings-benchmark/results`` repo). Task modalities come
from the mteb task registry.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mteb  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = Path(__file__).parent


def load_mteb_modalities() -> dict[str, tuple[str, ...]]:
    """{task_name: modalities} from the mteb task registry."""
    return {
        t.metadata.name: tuple(t.metadata.modalities)
        for t in mteb.get_tasks(modalities=["video"], exclude_beta=False)
    }


MTEB_MODALITIES: dict[str, tuple[str, ...]] = load_mteb_modalities()


def modality_label(task_name: str) -> str | None:
    """'audio + video' / 'video' / 'audio' header label (drops text)."""
    mods = MTEB_MODALITIES.get(task_name)
    if mods is None:
        return None
    data_mods = tuple(m for m in mods if m != "text")
    has_a = "audio" in data_mods
    has_v = "video" in data_mods
    if has_a and has_v:
        return "audio + video"
    if has_v:
        return "video"
    if has_a:
        return "audio"
    return None


# ---------------------------------------------------------------------------
# Model list & aliases
# ---------------------------------------------------------------------------

ROW_RX = re.compile(
    r"^(?P<name>[A-Za-z0-9_./\-]+)\s*(?:\\cite\{[^}]+\})?\s*&\s*[\d.]+\s*&\s*(?P<mods>[a-z, ]+?)\s*\\\\\s*$"
)

REFERENCE_MODEL_FOR_TASK_SET = "Haon-Chen/e5-omni-7B"


def parse_models(all_models_tex: Path) -> list[tuple[str, str]]:
    """Return list of (model_name, modalities_string) from all_models.tex."""
    out = []
    for ln in all_models_tex.read_text().splitlines():
        m = ROW_RX.match(ln.strip())
        if m:
            mods = ",".join(sorted(p.strip() for p in m.group("mods").split(",")))
            out.append((m.group("name"), mods))
    return out


def model_display(name: str) -> str:
    return name


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

# more-specific suffixes first
FAMILY_RX: list[tuple[str, re.Pattern[str]]] = [
    ("retrieval", re.compile(r".+(A2V|AT2V|T2VA|T2V|V2A|V2T|VA2T|VT2A)(Retrieval)?$")),
    ("zero-shot", re.compile(r".+(ZeroShotClassification|ZeroShot|Zeroshot)$")),
    ("pair-cls", re.compile(r".+PairClassification$")),
    ("qa", re.compile(r".+CentricQA$")),
    ("clustering", re.compile(r".+Clustering$")),
    ("classification", re.compile(r".+Classification(\.V\d+)?$")),
]

DIRECTION_RX = re.compile(
    r"^(?P<dataset>.+?)(?P<dir>A2V|AT2V|T2VA|T2V|V2A|V2T|VA2T|VT2A)(Retrieval)?$"
)

EXPLICIT_FAMILY: dict[str, str] = {
    "HumanAnimalCartoonV": "classification",
    "HumanAnimalCartoonVA": "classification",
    "Kinetics400V": "classification",
    "Kinetics400VA": "classification",
    "Kinetics600V": "classification",
    "Kinetics600VA": "classification",
    "Kinetics700V": "classification",
    "Kinetics700VA": "classification",
    "VGGSoundV": "classification",
    "VGGSoundVA": "classification",
}

SKIP_TASK_NAMES = {"model_meta"}

DIR_LABEL = {
    "T2V": r"Text $\to$ Video",
    "V2T": r"Video $\to$ Text",
    "A2V": r"Audio $\to$ Video",
    "V2A": r"Video $\to$ Audio",
    "AT2V": r"Audio+Text $\to$ Video",
    "T2VA": r"Text $\to$ Video+Audio",
    "VA2T": r"Video+Audio $\to$ Text",
    "VT2A": r"Video+Text $\to$ Audio",
}
# grouped by retrieval target: video, text, audio, video+audio
DIR_ORDER = ["T2V", "A2V", "AT2V", "V2T", "VA2T", "V2A", "VT2A", "T2VA"]


def task_family(name: str) -> str | None:
    if name in SKIP_TASK_NAMES:
        return None
    if name in EXPLICIT_FAMILY:
        return EXPLICIT_FAMILY[name]
    for fam, rx in FAMILY_RX:
        if rx.match(name):
            return fam
    return None


LEAF_SUFFIX_STRIPS = [
    "ZeroShotClassification",
    "PairClassification",
    "Classification.V2",
    "Classification",
    "CentricQA",
    "Clustering",
    "ZeroShot",
    "Zeroshot",
]


def short_task(t: str) -> str:
    m = DIRECTION_RX.match(t)
    if m:
        return m.group("dataset")
    for s in LEAF_SUFFIX_STRIPS:
        if t.endswith(s):
            stripped = t[: -len(s)]
            return stripped or t
    return t


# ---------------------------------------------------------------------------
# Loading scores
# ---------------------------------------------------------------------------


def load_score_table(cache: mteb.ResultCache) -> pd.DataFrame:
    """Per-task (rows) × per-model (cols) main_score table from the cache."""
    tasks = list(mteb.get_tasks(modalities=["video"], exclude_beta=False))
    return (
        cache.load_results(tasks=tasks, require_model_meta=False)
        .to_dataframe()
        .set_index("task_name")
    )


def _model_tasks(scores: pd.DataFrame, name: str) -> dict[str, float]:
    """Return {task: main_score} for a model, restricted to known families."""
    if name not in scores.columns:
        return {}
    out: dict[str, float] = {}
    for task, v in scores[name].items():
        if pd.isna(v) or task_family(task) is None:
            continue
        out[task] = float(v)
    return out


def load_scores(
    scores: pd.DataFrame, model_names: list[str], task_filter: set[str]
) -> dict[str, dict[str, float]]:
    """Load each model's scores, restricted to tasks in task_filter."""
    return {
        name: {t: v for t, v in _model_tasks(scores, name).items() if t in task_filter}
        for name in model_names
    }


def canonical_task_set(scores: pd.DataFrame) -> set[str]:
    tasks = _model_tasks(scores, REFERENCE_MODEL_FOR_TASK_SET)
    if not tasks:
        raise RuntimeError(
            f"Reference model {REFERENCE_MODEL_FOR_TASK_SET} has no results — "
            "pick another full-coverage omni model in REFERENCE_MODEL_FOR_TASK_SET."
        )
    return set(tasks.keys())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.2f}"


def latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


# longest-prefix match wins
DATASET_ABBREV: dict[str, str] = {
    "ActivityNetCaptions": "ActivityNet",
    "HumanAnimalCartoon": "HumAnimCartoon",
    "MusicAVQACLS": "MusicAVQA",
    "SomethingSomethingV2": "SSv2",
    "VideoMMEShort": "VideoMME-Short",
    "WorldSense1MinDomain": "WS-1m-Domain",
    "WorldSense1Min": "WS-1m",
}


def task_header(t: str) -> str:
    """LaTeX column header: dataset on line 1, direction-arrow (retrieval) or
    canonical mteb modality (other) on line 2."""
    # Retrieval: dataset on top, direction arrow on bottom.
    m = DIRECTION_RX.match(t)
    if m:
        ds = dataset_display(t)
        return rf"\makecell[c]{{{latex_escape(ds)} \\ {DIR_HEADER[m.group('dir')]}}}"

    # Non-retrieval: strip the explicit Video/AudioVideo/VideoAudio/bare-V suffix
    # from the dataset name (it's noise now that mteb gives us the canonical modality).
    base = short_task(t)
    for word in ("AudioVideo", "VideoAudio", "Video"):
        if base.endswith(word):
            base = base[: -len(word)]
            break
    else:
        for suffix in ("VA", "AV", "V"):
            if (
                base.endswith(suffix)
                and len(base) > len(suffix)
                and base[-len(suffix) - 1].isalnum()
            ):
                base = base[: -len(suffix)]
                break

    # apply dataset abbreviation (longest-prefix match)
    for orig in sorted(DATASET_ABBREV, key=len, reverse=True):
        if base.startswith(orig):
            base = DATASET_ABBREV[orig] + base[len(orig) :]
            break

    base = latex_escape(base.strip())
    second_line = modality_label(t) or r"\strut"
    return rf"\makecell[c]{{{base} \\ {second_line}}}"


# ---------------------------------------------------------------------------
# Table emission
# ---------------------------------------------------------------------------


HEADER_COMMENT = "% AUTOGENERATED by scripts/mveb_paper/gen_mveb_tex_tables.py — do not edit by hand.\n"


def _emit_single_block(
    lines: list[str],
    *,
    caption: str,
    label: str,
    header_tasks: list[str],
    model_order: list[str],
    scores: dict[str, dict[str, float]],
    per_model_avg: dict[str, float | None],
    best_per_col: dict[str, set[str]],
    best_avg: set[str],
) -> None:
    """Append one portrait table block (table*) covering header_tasks + Avg."""
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    col_spec = "l" + "c" * (len(header_tasks) + 1)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    hdr = [r"\textbf{Model}"]
    for t in header_tasks:
        hdr.append(task_header(t))
    hdr.append(r"\textbf{Avg.}")
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    for m in model_order:
        row = [latex_escape(model_display(m))]
        for t in header_tasks:
            v = scores[m].get(t)
            s = fmt(v)
            if v is not None and m in best_per_col.get(t, set()):
                s = bold(s)
            row.append(s)
        v = per_model_avg[m]
        s = fmt(v)
        if v is not None and m in best_avg:
            s = bold(s)
        row.append(s)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table*}")


def emit_family_table(
    path: Path,
    *,
    caption: str,
    label: str,
    tasks: list[str],
    models: list[str],
    scores: dict[str, dict[str, float]],
    split_into_two: bool = False,
) -> None:
    """models = rows, tasks = horizontal headers.

    When ``split_into_two`` is True, emit two portrait tables in the same
    file (alphabetical halves of ``tasks``). The Avg. column shown in each
    half is the full-family mean so models remain comparable across halves.
    """
    model_order = [
        m for m in models if any(scores[m].get(t) is not None for t in tasks)
    ]
    per_model_avg: dict[str, float | None] = {}
    for m in model_order:
        vals = [scores[m].get(t) for t in tasks]
        valid = [v for v in vals if v is not None]
        per_model_avg[m] = sum(valid) / len(valid) if valid else None

    best_per_col: dict[str, set[str]] = {}
    for t in tasks:
        vals = [(m, scores[m].get(t)) for m in model_order]
        valid = [(m, v) for m, v in vals if v is not None]
        if valid:
            mx = max(v for _, v in valid)
            best_per_col[t] = {m for m, v in valid if abs(v - mx) < 1e-9}

    valid_avg = [
        (m, per_model_avg[m]) for m in model_order if per_model_avg[m] is not None
    ]
    best_avg: set[str] = set()
    if valid_avg:
        mx = max(v for _, v in valid_avg)
        best_avg = {m for m, v in valid_avg if abs(v - mx) < 1e-9}

    lines: list[str] = [HEADER_COMMENT.rstrip()]

    if split_into_two:
        half = (len(tasks) + 1) // 2
        parts = [
            ("a", tasks[:half], "part 1 of 2"),
            ("b", tasks[half:], "part 2 of 2"),
        ]
        for suffix, part_tasks, part_label in parts:
            _emit_single_block(
                lines,
                caption=caption.rstrip(".") + f", {part_label}.",
                label=f"{label}-{suffix}",
                header_tasks=part_tasks,
                model_order=model_order,
                scores=scores,
                per_model_avg=per_model_avg,
                best_per_col=best_per_col,
                best_avg=best_avg,
            )
    else:
        _emit_single_block(
            lines,
            caption=caption,
            label=label,
            header_tasks=tasks,
            model_order=model_order,
            scores=scores,
            per_model_avg=per_model_avg,
            best_per_col=best_per_col,
            best_avg=best_avg,
        )

    path.write_text("\n".join(lines) + "\n")


# Compact arrow notation used in the mega-table's second header row.
DIR_HEADER = {
    "T2V": r"T$\to$V",
    "V2T": r"V$\to$T",
    "A2V": r"A$\to$V",
    "V2A": r"V$\to$A",
    "AT2V": r"AT$\to$V",
    "T2VA": r"T$\to$VA",
    "VA2T": r"VA$\to$T",
    "VT2A": r"VT$\to$A",
}


def dataset_display(t: str) -> str:
    """Short dataset name for a retrieval task (strips direction, applies abbrev)."""
    m = DIRECTION_RX.match(t)
    base = m.group("dataset") if m else t
    for orig in sorted(DATASET_ABBREV, key=len, reverse=True):
        if base.startswith(orig):
            return DATASET_ABBREV[orig] + base[len(orig) :]
    return base


def main(cache: mteb.ResultCache, out_dir: Path, models_tex: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    score_table = load_score_table(cache)
    all_models = parse_models(models_tex)
    model_names = [name for name, _ in all_models]
    canonical = canonical_task_set(score_table)
    scores = load_scores(score_table, model_names, canonical)
    models = [m for m in model_names if scores[m]]

    by_family: dict[str, list[str]] = defaultdict(list)
    for t in sorted(canonical):
        by_family[task_family(t)].append(t)

    retrieval_by_dir: dict[str, list[str]] = defaultdict(list)
    for t in by_family.get("retrieval", []):
        m = DIRECTION_RX.match(t)
        if m:
            retrieval_by_dir[m.group("dir")].append(t)

    def _caption(title: str, metric: str, n_tasks: int) -> str:
        return rf"\textbf{{{title}}} ({n_tasks} tasks, {metric})."

    for d in DIR_ORDER:
        if d not in retrieval_by_dir:
            continue
        tasks = sorted(retrieval_by_dir[d])
        emit_family_table(
            out_dir / f"results_retrieval_{d.lower()}.tex",
            caption=_caption(f"Retrieval: {DIR_LABEL[d]}", "nDCG@10", len(tasks)),
            label=f"tab:results-retrieval-{d.lower()}",
            tasks=tasks,
            models=models,
            scores=scores,
        )

    family_specs = {
        "classification": ("Classification", "tab:results-classification", "accuracy"),
        "zero-shot": ("Zero-shot classification", "tab:results-zero-shot", "accuracy"),
        "clustering": ("Clustering", "tab:results-clustering", "v-measure"),
        "pair-cls": ("Pair classification", "tab:results-pair-cls", "max-AP"),
        "qa": ("Question answering", "tab:results-qa", "accuracy"),
    }
    SPLIT_FAMILIES = {"classification", "zero-shot", "clustering", "pair-cls", "qa"}
    for fam, (title, label, metric) in family_specs.items():
        tasks = sorted(by_family.get(fam, []))
        if not tasks:
            continue
        emit_family_table(
            out_dir / f"results_{fam.replace('-', '_')}.tex",
            split_into_two=fam in SPLIT_FAMILIES,
            caption=_caption(title, metric, len(tasks)),
            label=label,
            tasks=tasks,
            models=models,
            scores=scores,
        )

    print(f"Wrote tables to {out_dir}")
    print(f"Models included: {len(models)}")
    for fam in (
        "retrieval",
        "classification",
        "zero-shot",
        "clustering",
        "pair-cls",
        "qa",
    ):
        print(f"  {fam:>15}: {len(by_family.get(fam, []))} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Path to a results cache directory (default: ~/.cache/mteb).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force-refresh the remote results repo before loading "
        "(otherwise the local cache is used, downloading only if absent).",
    )
    parser.add_argument(
        "--models-tex",
        required=True,
        help="Path to the paper's all_models.tex (the curated model roster, "
        "used for model ordering and modality annotations).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory the per-task .tex tables are written to (default: this "
        "script's folder). Point at the paper's tables/ directory to write there.",
    )
    args = parser.parse_args()
    cache = mteb.ResultCache(cache_path=args.cache_path)
    if args.download or not cache.has_remote:
        cache.download_from_remote()
    main(cache=cache, out_dir=Path(args.out_dir), models_tex=Path(args.models_tex))
