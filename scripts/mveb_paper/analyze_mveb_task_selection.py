#!/usr/bin/env python3
"""Generate MVEB task selection per modality scope.

Pipeline:
  1. Filter MVEB(extended) by scope (`audio-video`, `video-text`, `video`).
  2. Drop tasks whose audio use is invalid given the dataset's annotation
     provenance (e.g. audio-conditioned retrieval on MSRVTT).
  3. Drop saturated / floor / low-support tasks.
  4. Apply T2V-direction preference, family caps, and ρ-based redundancy
     pruning, with `MUST_INCLUDE` tasks bypassing every filter above.

Writes a markdown report per scope and prints a summary to stdout.

Usage:
    python scripts/mveb_paper/analyze_mveb_task_selection.py --scope audio-video \\
        --results-dir /path/to/results/results
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("mteb").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import mteb
from mteb.cache import ResultCache


# Correlation thresholds we sweep to report selection at each cut. The
# recommended operating point is 0.85.
THRESHOLDS = [
    0.99,
    0.98,
    0.97,
    0.96,
    0.95,
    0.93,
    0.9,
    0.88,
    0.87,
    0.85,
    0.84,
    0.83,
    0.82,
    0.81,
    0.8,
    0.7,
    0.6,
    0.5,
]

# Reference models used to estimate eval time of the selected subset.
EVAL_TIME_MODELS = [
    ("ebind-av", "encord-team/ebind-audio-vision"),  # 764M (smallest AV)
    ("pe-av-small", "facebook/pe-av-small"),
    ("LCO-Embedding-Omni-7B", "LCO-Embedding/LCO-Embedding-Omni-7B"),
    ("Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-7B"),
]

# Manually protected (the only sport-domain representative).
MANUALLY_PROTECTED_TASKS = ["Diving48Classification.V1"]

TASKS_TO_EXCLUDE: list[str] = []


# Datasets where audio carries label-relevant signal: AV-aware captions,
# AV-required QA, emotion (prosody), or action classes whose audio is
# typically informative.
AV_AWARE_FAMILIES = {
    "VALOR32K",
    "AudioCapsAV",
    "Shot2Story20K",
    "VGGSoundAV",
    "AVEDataset",
    "MELD",
    "RAVDESS",
    "MusicAVQA",
    "MusicAVQACLS",
    "AVQA",
    "AVMeme",
    "AVMemeExam",
    "WorldSense",
    "WorldSense1Min",
    "DailyOmni",
    "OmniVideoBench",
    "VideoMME",
    "VideoMMEShort",
    "PerceptionTest",
    "AVSpeakerBench",
    "Kinetics400",
    "Kinetics600",
    "Kinetics700",
    "UCF101",
    "HMDB51",
    "Breakfast",
    "HumanAnimalCartoon",
    "VATEX",
    "ActivityNetCaptions",
    "YouCook2",
}

# Datasets where labels describe visual content only — audio on these is
# incidental and would not match the label.
VIDEO_ONLY_FAMILIES = {
    "MSRVTT",
    "MSVD",
    "DiDeMo",
    "Panda70M",
    "TUNABench",
    "Charades-STA",
    "SomethingSomethingV2",
    "Diving48",
    "Vinoground",
    "VideoCon",
    "NExTQA",
    "EgoSchema",
}


# Tasks that must end up in the final benchmark — either standard datasets
# the field expects, V-only variants needed for the nested MVEB(video) subset,
# or AV-joint variants needed for AV task-type coverage in the master.
MUST_INCLUDE: set[str] = set()
#     # Retrieval — standard T2V/V2T canonicals:
#     "MSRVTTT2V", "MSRVTTV2T", "MSVDT2VRetrieval", "VATEXT2VRetrieval",
#     "DiDeMoT2VRetrieval", "YouCook2T2VRetrieval",
#     "ActivityNetCaptionsT2VRetrieval",
#     # Retrieval — audio-conditioned on AV-aware families:
#     "AudioCapsAVAT2VRetrieval", "AudioCapsAVVA2TRetrieval",
#     "VALOR32KT2VARetrieval", "VALOR32KVT2ARetrieval", "VALOR32KA2VRetrieval",
#     "AVMemeExamAT2VRetrieval",
#     # Classification — standard action recognition + V-only variants for V subset:
#     "HMDB51Classification", "BreakfastClassification",
#     "SomethingSomethingV2Classification", "VGGSoundVA",
#     "AVEDatasetClassification", "AVMemeVideoClassification",
#     "MELDVideoClassification", "WorldSenseVideoClassification",
#     "AVMemeAudioVideoClassification",
#     # Clustering — V variants for V subset + AV variants for master:
#     "AVEDatasetVideoClustering", "RAVDESSVideoClustering",
#     "AVEDatasetAudioVideoClustering", "WorldSense1MinDomainAudioVideoClustering",
#     # QA:
#     "EgoSchemaVideoCentricQA", "NExTQAVideoCentricQA",
#     "VideoMMEShortVideoCentricQA", "OmniVideoBenchVideoCentricQA",
#     "WorldSense1MinVideoAudioCentricQA", "DailyOmniVideoAudioCentricQA",
#     # Pair-cls:
#     "VinogroundPairClassification", "RAVDESSAVVAPairClassification",
#     "HumanAnimalCartoonVPairClassification",
#     # Zero-shot:
#     "HMDB51ZeroShot", "UCF101VideoZeroShotClassification", "MELDVideoZeroShot",
#     "WorldSenseAudioVideoZeroShot",
# }


# Discriminative-power filter thresholds.
SAT_BEST_THRESHOLD = 0.93
FLOOR_SPREAD_THRESHOLD = 0.05
MIN_MODEL_SUPPORT = 3


# Each scope keeps tasks whose modalities ⊆ `allowed_task_modalities`.
SCOPES = {
    "video": {
        "description": "MVEB(video) — V-only encoders (vjepa2, etc.)",
        "allowed_task_modalities": {"video"},
    },
    "video-text": {
        "description": "MVEB(text, video) — T+V encoders (xclip, UME-R1, ebind-points-vision, +)",
        "allowed_task_modalities": {"video", "text", "image"},
    },
    "audio-video": {
        "description": "MVEB — full A+V+T encoders (pe-av, ebind-av, omni, +)",
        "allowed_task_modalities": {"audio", "video", "text", "image"},
    },
}

# Retrieval task families - for each family, prefer T2V over V2T
# If both V2TRetrieval and T2VRetrieval exist, remove V2T
RETRIEVAL_FAMILIES = [
    "MSVD",
    "TUNABench",
    "VATEX",
    "Panda70M",
    "YouCook2",
    "Shot2Story20K",
    "VALOR32K",
    "DiDeMo",
    "MSRVTT",
    "ActivityNetCaptions",
    "AudioCapsAV",
    "AVMemeExam",
    "VGGSoundAV",
]

# Same-source families for deduplication
# Tasks from the same family and same task type are considered redundant
SAME_SOURCE_FAMILIES = [
    "MELD",  # Classification, Clustering, ZeroShot (audio-video and video-only)
    "WorldSense",  # Classification, Clustering, QA variants
    "AVEDataset",  # Classification, Clustering, ZeroShot
    "AVMeme",  # Classification, ZeroShot, QA variants
    "MusicAVQACLS",  # Classification, Clustering, ZeroShot
    "RAVDESS",  # Classification, Clustering, ZeroShot
    "UCF101",  # Classification, Clustering, ZeroShot
    "HumanAnimalCartoon",  # Classification, ZeroShot
    "Kinetics400",  # Classification, ZeroShot
    "Kinetics600",  # Classification, ZeroShot
    "Kinetics700",  # Classification, ZeroShot
    "VGGSound",  # Classification, ZeroShot, Retrieval
    "HMDB51",  # Classification, Clustering, ZeroShot
    "Breakfast",  # Classification, ZeroShot
]


def family_of(task_name: str) -> str | None:
    """Return the family prefix this task belongs to, or None.

    Longer prefixes are checked first so e.g. "MusicAVQACLS" wins over "MusicAVQA".
    """
    families = sorted(AV_AWARE_FAMILIES | VIDEO_ONLY_FAMILIES, key=len, reverse=True)
    for fam in families:
        if task_name.startswith(fam):
            return fam
    return None


def is_annotation_valid(task_name: str, metadata_df: pd.DataFrame) -> tuple[bool, str]:
    """Whether using audio on this task is principled given the dataset's
    annotation protocol. Audio is only fair on AV-aware families.
    """
    fam = family_of(task_name)
    if fam is None or fam not in VIDEO_ONLY_FAMILIES:
        return True, ""

    rows = metadata_df[metadata_df["name"] == task_name]
    if not rows.empty and "audio" in set(rows.iloc[0]["modalities"]):
        return False, f"uses audio but '{fam}' has visual-only labels"
    return True, ""


def filter_invalid_annotation(
    tasks: list[str], protected: set[str], metadata_df: pd.DataFrame
) -> tuple[list[str], list[tuple[str, str]]]:
    kept, dropped = [], []
    for t in tasks:
        if t in protected:
            kept.append(t)
            continue
        valid, reason = is_annotation_valid(t, metadata_df)
        if valid:
            kept.append(t)
        else:
            dropped.append((t, reason))
    return kept, dropped


def filter_by_scope(
    tasks: list[str], metadata_df: pd.DataFrame, scope_key: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep tasks whose modalities ⊆ scope's allowed set."""
    allowed = SCOPES[scope_key]["allowed_task_modalities"]
    kept, dropped = [], []
    for t in tasks:
        rows = metadata_df[metadata_df["name"] == t]
        if rows.empty:
            dropped.append((t, "no metadata"))
            continue
        task_mods = set(rows.iloc[0]["modalities"])
        if task_mods.issubset(allowed):
            kept.append(t)
        else:
            dropped.append(
                (t, f"modalities {sorted(task_mods - allowed)} outside scope")
            )
    return kept, dropped


def prefer_av_variants(
    tasks: list[str], metadata_df: pd.DataFrame, scope_key: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """For the audio-video scope, if a family has both AV variants (tasks with
    audio in their modalities) and non-AV variants (no audio), drop the non-AV
    variants so that task selection focuses on genuinely multimodal tasks."""
    if scope_key != "audio-video":
        return tasks, []

    all_families = sorted(
        AV_AWARE_FAMILIES | VIDEO_ONLY_FAMILIES, key=len, reverse=True
    )

    def _family(name: str) -> str | None:
        for fam in all_families:
            if name.startswith(fam):
                return fam
        return None

    # Group tasks by family
    family_tasks: dict[str, list[str]] = defaultdict(list)
    no_family: list[str] = []
    for t in tasks:
        fam = _family(t)
        if fam:
            family_tasks[fam].append(t)
        else:
            no_family.append(t)

    kept, dropped = list(no_family), []
    for fam, members in family_tasks.items():
        av_members = []
        non_av_members = []
        for t in members:
            rows = metadata_df[metadata_df["name"] == t]
            if rows.empty:
                av_members.append(t)  # keep if unknown
                continue
            task_mods = set(rows.iloc[0]["modalities"])
            if "audio" in task_mods:
                av_members.append(t)
            else:
                non_av_members.append(t)

        if av_members:
            # Family has AV variants — keep only those
            kept.extend(av_members)
            for t in non_av_members:
                dropped.append(
                    (t, f"non-AV variant dropped; family '{fam}' has AV variants")
                )
        else:
            # No AV variants exist — keep all
            kept.extend(non_av_members)

    # Preserve original ordering
    task_order = {t: i for i, t in enumerate(tasks)}
    kept.sort(key=lambda t: task_order.get(t, 0))
    return kept, dropped


def load_model_modalities(results_dir: Path | str) -> dict[str, set[str]]:
    """{model_name: modalities_set} read from each model's model_meta.json."""
    results_dir = Path(results_dir)
    out: dict[str, set[str]] = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        revs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not revs:
            continue
        revs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        meta_path = revs[0] / "model_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            out[model_dir.name.replace("__", "/")] = set(meta.get("modalities") or [])
        except (json.JSONDecodeError, OSError):
            continue
    return out


def compute_task_stats(
    results_df: pd.DataFrame,
    tasks: list[str],
    metadata_df: pd.DataFrame,
    model_modalities: dict[str, set[str]],
) -> dict[str, dict]:
    """{task: {best, worst, spread, n}} across models whose modalities ⊇ task's."""
    empty = {"best": None, "worst": None, "spread": None, "n": 0}
    stats: dict[str, dict] = {}
    for t in tasks:
        if t not in results_df.columns:
            stats[t] = dict(empty)
            continue
        rows = metadata_df[metadata_df["name"] == t]
        if rows.empty:
            stats[t] = dict(empty)
            continue
        task_mods = set(rows.iloc[0]["modalities"])
        capable = [
            m for m, mods in model_modalities.items() if task_mods.issubset(mods)
        ]
        capable_in_df = [m for m in capable if m in results_df.index]
        scores = (
            results_df.loc[capable_in_df, t].dropna()
            if capable_in_df
            else pd.Series(dtype=float)
        )
        if scores.empty:
            stats[t] = dict(empty)
            continue
        stats[t] = {
            "best": float(scores.max()),
            "worst": float(scores.min()),
            "spread": float(scores.max() - scores.min()),
            "n": int(len(scores)),
        }
    return stats


def filter_saturation_floor(
    tasks: list[str],
    stats: dict[str, dict],
    protected: set[str],
    sat_threshold: float = SAT_BEST_THRESHOLD,
    floor_spread: float = FLOOR_SPREAD_THRESHOLD,
    min_support: int = MIN_MODEL_SUPPORT,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Drop saturated / floor / low-support tasks. Protected tasks bypass."""
    kept, dropped = [], []
    for t in tasks:
        if t in protected:
            kept.append(t)
            continue
        s = stats.get(t)
        if s is None or s.get("best") is None:
            dropped.append((t, "no model results from capable models"))
            continue
        if s["n"] < min_support:
            dropped.append(
                (t, f"low support (only {s['n']} capable models with results)")
            )
            continue
        if s["best"] > sat_threshold:
            dropped.append((t, f"saturated (best={s['best']:.3f} > {sat_threshold})"))
            continue
        if s["spread"] < floor_spread:
            dropped.append((t, f"floor (spread={s['spread']:.3f} < {floor_spread})"))
            continue
        kept.append(t)
    return kept, dropped


def deduplicate_retrieval_directions(
    task_names: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """For each family that has both T2V and V2T, drop V2T (T2V is the
    more commonly reported direction)."""
    v2t_tasks: dict[str, str] = {}
    t2v_tasks: dict[str, str] = {}
    for task in task_names:
        for family in RETRIEVAL_FAMILIES:
            if task.startswith(family):
                if "V2TRetrieval" in task:
                    v2t_tasks[family] = task
                elif "T2VRetrieval" in task:
                    t2v_tasks[family] = task
                break

    tasks_to_remove = set()
    removed = []
    for family, v2t_task in v2t_tasks.items():
        if family in t2v_tasks:
            tasks_to_remove.add(v2t_task)
            removed.append(
                (
                    v2t_task,
                    f"Prefer T2V over V2T for {family} (keeping {t2v_tasks[family]})",
                )
            )

    remaining = [t for t in task_names if t not in tasks_to_remove]
    return remaining, removed


def enforce_retrieval_direction_preference(
    task_names: list[str],
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Same as `deduplicate_retrieval_directions` but returns the
    (task, reason, correlation) tuple shape used by the selection pipeline."""
    remaining, removed_pairs = deduplicate_retrieval_directions(task_names)
    removed = [(task, reason, 0.0) for task, reason in removed_pairs]
    return remaining, removed


def deduplicate_same_source_families(
    task_names: list[str],
    results_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    protected_tasks: set[str] | None = None,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Cap tasks per source dataset family in two passes:

    1. Within (family, task_type): keep one task (lowest avg ρ to the rest).
    2. Across non-retrieval task types within a family: keep one task.
       Retrieval is exempt — different directions (T2V, A2V, ...) from
       the same dataset are distinct measurements.

    Protected tasks always survive; if two protected tasks fall in the same
    group, both are kept.
    """
    remaining = task_names.copy()
    removed = []
    protected_tasks = set(protected_tasks) if protected_tasks else set()

    task_to_family = {}
    for task in task_names:
        for family in SAME_SOURCE_FAMILIES:
            if task.startswith(family) or family.lower() in task.lower():
                task_rows = metadata_df[metadata_df["name"] == task]
                if len(task_rows) > 0:
                    task_meta = task_rows.iloc[0]
                    task_to_family[task] = (family, task_meta["type"])
                break

    def pick_lowest_corr(candidates: list[str]) -> str | None:
        """The candidate with the lowest mean |ρ| against the rest of `remaining`."""
        best = None
        best_avg = float("inf")
        for cand in candidates:
            if cand not in results_df.columns:
                continue
            others = [t for t in remaining if t != cand and t in results_df.columns]
            if not others:
                continue
            corrs = []
            for o in others:
                c = results_df[[cand, o]].corr(method="spearman").iloc[0, 1]
                if not np.isnan(c):
                    corrs.append(abs(c))
            if corrs:
                avg = np.mean(corrs)
                if avg < best_avg:
                    best_avg = avg
                    best = cand
        return best

    # Pass 1: dedup within (family, task_type)
    family_type_groups = defaultdict(list)
    for task, (family, task_type) in task_to_family.items():
        family_type_groups[(family, task_type)].append(task)

    for (family, task_type), tasks in family_type_groups.items():
        if len(tasks) <= 1:
            continue
        protected_here = [t for t in tasks if t in protected_tasks]
        # If any protected tasks are in this group, keep all of them; only
        # consider removing non-protected siblings.
        if protected_here:
            for t in tasks:
                if t not in protected_tasks and t in remaining:
                    remaining.remove(t)
                    removed.append(
                        (
                            t,
                            f"Same-family ({family}) same-type ({task_type}) "
                            f"redundancy, keeping protected {protected_here}",
                            0.0,
                        )
                    )
            continue
        keep = pick_lowest_corr(tasks)
        if keep:
            for t in tasks:
                if t != keep and t in remaining:
                    remaining.remove(t)
                    removed.append(
                        (
                            t,
                            f"Same-family ({family}) same-type ({task_type}) redundancy, keeping {keep}",
                            0.0,
                        )
                    )

    # Pass 2: cap to 1 non-retrieval task per family (retrieval directions
    # from the same dataset are distinct measurements, not redundant).
    family_nonret_groups = defaultdict(list)
    for task in remaining:
        if task not in task_to_family:
            continue
        family, task_type = task_to_family[task]
        if task_type and "retrieval" in task_type.lower():
            continue
        family_nonret_groups[family].append(task)

    for family, tasks in family_nonret_groups.items():
        if len(tasks) <= 1:
            continue
        protected_here = [t for t in tasks if t in protected_tasks]
        if protected_here:
            # Keep all protected; remove only non-protected siblings.
            for t in tasks:
                if t not in protected_tasks and t in remaining:
                    remaining.remove(t)
                    removed.append(
                        (
                            t,
                            f"Same-family ({family}) non-retrieval cap, "
                            f"keeping protected {protected_here}",
                            0.0,
                        )
                    )
            continue
        keep = pick_lowest_corr(tasks)
        if keep:
            for t in tasks:
                if t != keep and t in remaining:
                    remaining.remove(t)
                    removed.append(
                        (
                            t,
                            f"Same-family ({family}) non-retrieval cap, keeping {keep}",
                            0.0,
                        )
                    )

    return remaining, removed


def load_eval_times(
    results_dir: Path, model_name: str, task_names: list[str]
) -> dict[str, float]:
    """Read `evaluation_time` (seconds) from each task's result JSON for `model_name`."""
    model_folder = results_dir / model_name.replace("/", "__").replace(" ", "_")
    if not model_folder.exists():
        return {}

    revs = sorted(
        [f for f in model_folder.iterdir() if f.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not revs:
        return {}

    eval_times: dict[str, float] = {}
    for task_name in task_names:
        task_file = revs[0] / f"{task_name}.json"
        if not task_file.exists():
            continue
        try:
            data = json.loads(task_file.read_text())
            if data.get("evaluation_time") is not None:
                eval_times[task_name] = data["evaluation_time"]
        except json.JSONDecodeError:
            pass
    return eval_times


def compute_total_eval_time(
    eval_times: dict[str, float], task_names: list[str]
) -> tuple[float, int]:
    """Sum eval times across `task_names`. Returns (total_seconds, tasks_with_data)."""
    total = sum(eval_times[t] for t in task_names if t in eval_times)
    count = sum(1 for t in task_names if t in eval_times)
    return total, count


def format_duration(seconds: float) -> str:
    """Format `seconds` as "1h 23m" or "45m 30s"."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {int(seconds % 60)}s"


def get_task_metadata_df(benchmark_name: str) -> pd.DataFrame:
    """Build a DataFrame of task metadata for the given benchmark."""
    benchmark = mteb.get_benchmark(benchmark_name)

    rows = []
    for task in benchmark.tasks:
        meta = task.metadata
        rows.append(
            {
                "name": meta.name,
                "languages": tuple(sorted(meta.languages)) if meta.languages else (),
                "domains": tuple(sorted(meta.domains)) if meta.domains else (),
                "type": meta.type,
                "category": meta.category,
                "dataset_path": meta.dataset.get("path", ""),
                "modalities": tuple(sorted(meta.modalities)),
            }
        )

    return pd.DataFrame(rows)


def identify_task_families(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group tasks by their source-dataset prefix (e.g. UCF101 → UCF101Classification,
    UCF101Clustering, UCF101ZeroshotClassification). Returns families with >1 task."""
    families = defaultdict(list)
    common_prefixes = [
        "MSVD",
        "TUNABench",
        "VATEX",
        "Panda70M",
        "YouCook2",
        "Shot2Story20K",
        "VALOR32K",
        "DiDeMo",
        "MSRVTT",
        "ActivityNetCaptions",
        "AudioCapsAV",
        "AVMemeExam",
        "VGGSoundAV",
        "Kinetics400",
        "Kinetics600",
        "Kinetics700",
        "UCF101",
        "HMDB51",
        "MELD",
        "WorldSense",
        "AVEDataset",
        "MusicAVQACLS",
        "RAVDESS",
        "HumanAnimalCartoon",
        "Breakfast",
        "VGGSound",
        "AVMeme",
        "Diving48",
    ]
    for task_name in df["name"].tolist():
        matched = False
        for prefix in common_prefixes:
            if task_name.startswith(prefix) or prefix.lower() in task_name.lower():
                families[prefix].append(task_name)
                matched = True
                break
        if not matched:
            families["Other"].append(task_name)

    # Only return families with multiple tasks
    return {
        family: sorted(tasks) for family, tasks in families.items() if len(tasks) > 1
    }


def load_model_results(results_dir: Path | str) -> tuple[pd.DataFrame, list[str]]:
    """Load all video-task results via mteb's ResultCache.

    Returns (results_df indexed by model with tasks as columns, list_of_task_names).
    Drops models with > 10 NaN tasks so the correlation matrix isn't dominated
    by sparse rows.
    """
    results_dir = Path(results_dir)
    model_names = [
        d.name.replace("__", "/") for d in results_dir.iterdir() if d.is_dir()
    ]
    print(f"Found {len(model_names)} models")

    models = []
    for name in model_names:
        try:
            models.append(mteb.get_model_meta(name))
        except (KeyError, ValueError):
            pass
    print(f"Models with metadata: {len(models)}")

    # cache_path is the parent of results_dir; works for both
    # ~/.cache/mteb/remote/results and ~/repo/results/results layouts.
    cache = ResultCache(cache_path=str(results_dir.parent))
    video_tasks = mteb.get_tasks(modalities=["video"], exclude_beta=False)
    mteb_results = cache.load_results(
        models=models, tasks=video_tasks, require_model_meta=False
    )

    full_df = mteb_results.to_dataframe().set_index("task_name").T
    results_df = full_df[full_df.isna().sum(axis=1) <= 10]
    print(f"Models after NaN filter (<=10 NaN): {len(results_df)}")
    print(f"Tasks with results: {len(results_df.columns)}")
    return results_df, list(results_df.columns)


def compute_task_correlation(
    results_df: pd.DataFrame, tasks: list[str]
) -> pd.DataFrame:
    """Pairwise Spearman correlation across tasks (rows=models, cols=tasks)."""
    return results_df[tasks].select_dtypes(include=["number"]).corr(method="spearman")


def compute_benchmark_correlation(
    results_df: pd.DataFrame,
    source_tasks: list[str],
    selected_tasks: list[str],
) -> tuple[float, float]:
    """How well does the per-model mean over `selected_tasks` rank-correlate with
    the per-model mean over `source_tasks`. Returns (spearman, pearson) or NaN
    if fewer than 3 models have data on both."""
    source_available = [t for t in source_tasks if t in results_df.columns]
    selected_available = [t for t in selected_tasks if t in results_df.columns]
    if not source_available or not selected_available:
        return float("nan"), float("nan")

    source_avg = results_df[source_available].mean(axis=1)
    selected_avg = results_df[selected_available].mean(axis=1)
    mask = ~(source_avg.isna() | selected_avg.isna())
    source_avg, selected_avg = source_avg[mask], selected_avg[mask]
    if len(source_avg) < 3:
        return float("nan"), float("nan")

    spearman_corr, _ = spearmanr(source_avg, selected_avg)
    pearson_corr, _ = pearsonr(source_avg, selected_avg)
    return spearman_corr, pearson_corr


def get_coverage_analysis(task_names: list[str]) -> dict:
    """Analyze language/domain/category/type coverage for a set of tasks.

    Args:
        task_names: List of task names to analyze

    Returns:
        Dictionary with coverage statistics
    """
    tasks = mteb.get_tasks(tasks=task_names)

    languages = set()
    domains = set()
    categories = set()
    types = set()
    type_counts = Counter()
    category_counts = Counter()
    domain_counts = Counter()

    for task in tasks:
        meta = task.metadata
        if meta.languages:
            languages.update(meta.languages)
        if meta.domains:
            domains.update(meta.domains)
            domain_counts.update(meta.domains)
        if meta.category:
            categories.add(meta.category)
            category_counts[meta.category] += 1
        types.add(meta.type)
        type_counts[meta.type] += 1

    return {
        "n_tasks": len(task_names),
        "n_languages": len(languages),
        "n_domains": len(domains),
        "n_categories": len(categories),
        "n_types": len(types),
        "languages": sorted(languages),
        "domains": sorted(domains),
        "categories": sorted(categories),
        "types": sorted(types),
        "type_counts": dict(type_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "domain_counts": dict(domain_counts.most_common()),
    }


def get_highly_correlated_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """All (task1, task2, ρ) pairs with ρ above `threshold`, sorted descending."""
    cols = corr_matrix.columns.tolist()
    pairs = [
        (t1, t2, corr_matrix.loc[t1, t2])
        for i, t1 in enumerate(cols)
        for t2 in cols[i + 1 :]
        if not np.isnan(corr_matrix.loc[t1, t2]) and corr_matrix.loc[t1, t2] > threshold
    ]
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def get_unique_coverage_tasks(task_names: list[str]) -> dict[str, list[str]]:
    """For each of (language, domain, category, type), find tasks that are the
    only carrier of a given value. These are auto-protected from pruning."""
    tasks = mteb.get_tasks(tasks=task_names)
    lang_to_tasks: dict = defaultdict(list)
    domain_to_tasks: dict = defaultdict(list)
    category_to_tasks: dict = defaultdict(list)
    type_to_tasks: dict = defaultdict(list)

    for task in tasks:
        meta = task.metadata
        name = meta.name
        for lang in meta.languages or []:
            lang_to_tasks[lang].append(name)
        for domain in meta.domains or []:
            domain_to_tasks[domain].append(name)
        if meta.category:
            category_to_tasks[meta.category].append(name)
        type_to_tasks[meta.type].append(name)

    unique_lang_tasks = [(ts[0], k) for k, ts in lang_to_tasks.items() if len(ts) == 1]
    unique_domain_tasks = [
        (ts[0], k) for k, ts in domain_to_tasks.items() if len(ts) == 1
    ]
    unique_category_tasks = [
        (ts[0], k) for k, ts in category_to_tasks.items() if len(ts) == 1
    ]
    unique_type_tasks = [(ts[0], k) for k, ts in type_to_tasks.items() if len(ts) == 1]

    return {
        "unique_language": unique_lang_tasks,
        "unique_domain": unique_domain_tasks,
        "unique_category": unique_category_tasks,
        "unique_type": unique_type_tasks,
    }


def is_removal_valid(
    current_tasks: list[str],
    task_to_remove: str,
    protected_tasks: set[str],
) -> bool:
    """Check if removing a task would lose unique coverage.

    Args:
        current_tasks: Current list of tasks
        task_to_remove: Task being considered for removal
        protected_tasks: Tasks that must not be removed

    Returns:
        True if removal is valid (won't lose unique coverage)
    """
    if task_to_remove in protected_tasks:
        return False

    remaining_tasks = [t for t in current_tasks if t != task_to_remove]
    if not remaining_tasks:
        return False

    task = mteb.get_task(task_to_remove)
    remaining = mteb.get_tasks(tasks=remaining_tasks)

    # Check unique task type
    remaining_types = {t.metadata.type for t in remaining}
    if task.metadata.type not in remaining_types:
        return False

    # Check unique category
    if task.metadata.category:
        remaining_cats = {t.metadata.category for t in remaining if t.metadata.category}
        if task.metadata.category not in remaining_cats:
            return False

    # Check unique domain
    if task.metadata.domains:
        remaining_domains = set()
        for t in remaining:
            if t.metadata.domains:
                remaining_domains.update(t.metadata.domains)
        for domain in task.metadata.domains:
            if domain not in remaining_domains:
                return False

    # Check unique language
    if task.metadata.languages:
        remaining_langs = set()
        for t in remaining:
            if t.metadata.languages:
                remaining_langs.update(t.metadata.languages)
        for lang in task.metadata.languages:
            if lang not in remaining_langs:
                return False

    return True


def iterative_task_selection(
    results_df: pd.DataFrame,
    initial_tasks: list[str],
    metadata_df: pd.DataFrame,
    threshold: float = 0.8,
    protected_tasks: set[str] | None = None,
    prefer_remove_same_source: bool = True,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Iteratively drop the highest-correlated pair member whose removal still
    preserves coverage (language, domain, type, category) and isn't protected.
    Returns (remaining_tasks, [(removed_task, reason, ρ_at_removal), ...])."""
    protected_tasks = set(protected_tasks) if protected_tasks else set()
    current_tasks = initial_tasks.copy()
    removed_tasks = []
    task_families = identify_task_families(metadata_df)
    task_to_family = {m: f for f, members in task_families.items() for m in members}

    while True:
        available = [t for t in current_tasks if t in results_df.columns]
        if len(available) < 2:
            break
        corr_matrix = compute_task_correlation(results_df, available)
        pairs = get_highly_correlated_pairs(corr_matrix, threshold)
        if not pairs:
            break

        removed_this_round = False
        for task1, task2, corr_val in pairs:
            if task1 not in current_tasks or task2 not in current_tasks:
                continue
            meta1 = metadata_df[metadata_df["name"] == task1].iloc[0]
            meta2 = metadata_df[metadata_df["name"] == task2].iloc[0]

            def removal_priority(task_name: str, meta: pd.Series) -> float:
                """Higher = prefer to remove. V2T retrieval (when a T2V sibling
                exists) is the biggest signal — we always want T2V over V2T."""
                score = 0
                if "V2TRetrieval" in task_name:
                    for family in RETRIEVAL_FAMILIES:
                        if task_name.startswith(family):
                            if f"{family}T2VRetrieval" in current_tasks:
                                score += 100
                            break
                elif "T2VRetrieval" in task_name:
                    for family in RETRIEVAL_FAMILIES:
                        if task_name.startswith(family):
                            if f"{family}V2TRetrieval" in current_tasks:
                                score -= 100
                            break

                # Prefer removing v2t over t2v for retrieval (general preference)
                if meta["category"] == "v2t":
                    score += 2
                elif meta["category"] == "t2v":
                    score -= 1

                # Prefer removing clustering over classification
                if "Clustering" in task_name:
                    score += 1
                if "Classification" in task_name or "ID" in task_name:
                    score -= 0.5

                # Prefer removing redundant siblings from the same source family.
                family = task_to_family.get(task_name)
                if family and prefer_remove_same_source:
                    family_count = sum(
                        1 for t in current_tasks if task_to_family.get(t) == family
                    )
                    if family_count > 1:
                        score += family_count
                return score

            candidates = sorted(
                [
                    (task1, removal_priority(task1, meta1)),
                    (task2, removal_priority(task2, meta2)),
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            for task_name, _ in candidates:
                if is_removal_valid(current_tasks, task_name, protected_tasks):
                    current_tasks.remove(task_name)
                    other = task1 if task_name == task2 else task2
                    family = task_to_family.get(task_name, "")
                    reason = f"corr={corr_val:.3f} with {other}"
                    if family:
                        reason += f", family={family}"
                    removed_tasks.append((task_name, reason, corr_val))
                    removed_this_round = True
                    break
            if removed_this_round:
                break

        if not removed_this_round:
            break

    return current_tasks, removed_tasks


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=list(SCOPES.keys()),
        default="audio-video",
        help="Modality scope of the variant to select.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/Users/isaac/.cache/mteb/remote/results"),
        help="Path to the mteb results cache.",
    )
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=SAT_BEST_THRESHOLD,
        help="Drop if best capable-model score exceeds this.",
    )
    parser.add_argument(
        "--floor-spread",
        type=float,
        default=FLOOR_SPREAD_THRESHOLD,
        help="Drop if (max - min) across capable models is below this.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=MIN_MODEL_SUPPORT,
        help="Drop if fewer than this many capable models have results.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    scope_key = args.scope
    scope_meta = SCOPES[scope_key]
    print(f"Scope: {scope_key} — {scope_meta['description']}")

    mveb_extended = mteb.get_benchmark("MVEB(extended)")
    source_task_names = [t.metadata.name for t in mveb_extended.tasks]
    print(f"Source pool - MVEB(extended) tasks: {len(source_task_names)}")
    metadata_df = get_task_metadata_df("MVEB(extended)")

    results_dir = args.results_dir
    corr_matrix = None
    results_df = None
    if results_dir.exists():
        print("\nLoading model results for correlation analysis...")
        try:
            results_df, tasks_with_results = load_model_results(results_dir)
            available_tasks = [t for t in source_task_names if t in tasks_with_results]
            print(
                f"Tasks with results: {len(available_tasks)}/{len(source_task_names)}"
            )
            if available_tasks:
                corr_matrix = compute_task_correlation(results_df, available_tasks)
                print(
                    f"Computed {len(corr_matrix)}x{len(corr_matrix)} correlation matrix"
                )
        except Exception as e:
            print(f"Warning: Could not load model results: {e}")
            print("Continuing without correlation analysis...")
    else:
        print(f"\nResults directory not found: {results_dir}")
        print("Skipping correlation analysis...")

    if results_df is not None and corr_matrix is not None:
        # Drop tasks with no results — they would survive iterative pruning by
        # never appearing in correlation pairs.
        missing = [
            t
            for t in source_task_names
            if t not in TASKS_TO_EXCLUDE and t not in results_df.columns
        ]
        if missing:
            print(f"Note: {len(missing)} task(s) dropped (no model results): {missing}")
        filtered_source_tasks = [
            t
            for t in source_task_names
            if t not in TASKS_TO_EXCLUDE and t in results_df.columns
        ]
        excluded_count = sum(1 for t in source_task_names if t in TASKS_TO_EXCLUDE)

        # Pre-selection: scope → annotation → saturation. Then T2V-pref /
        # family-cap / ρ-prune runs on the survivors.
        scoped_tasks, scope_dropped = filter_by_scope(
            filtered_source_tasks, metadata_df, scope_key
        )
        print(f"Scope filter: {len(filtered_source_tasks)} → {len(scoped_tasks)}")

        scoped_tasks, av_pref_dropped = prefer_av_variants(
            scoped_tasks, metadata_df, scope_key
        )
        if av_pref_dropped:
            print(
                f"AV-variant preference: dropped {len(av_pref_dropped)} non-AV variants"
            )

        must_keep = MUST_INCLUDE & set(scoped_tasks)
        annotation_valid, annotation_dropped = filter_invalid_annotation(
            scoped_tasks, must_keep, metadata_df
        )
        print(f"Annotation filter: {len(scoped_tasks)} → {len(annotation_valid)}")

        model_modalities = load_model_modalities(results_dir)
        task_stats = compute_task_stats(
            results_df, annotation_valid, metadata_df, model_modalities
        )
        working_pool, sat_dropped = filter_saturation_floor(
            annotation_valid,
            task_stats,
            must_keep,
            sat_threshold=args.sat_threshold,
            floor_spread=args.floor_spread,
            min_support=args.min_support,
        )
        print(f"Saturation/floor filter: {len(annotation_valid)} → {len(working_pool)}")

        filtered_source_tasks = working_pool

        # Build output for markdown file
        output_lines = []
        output_lines.append(f"# MVEB Task Selection — scope: `{scope_key}`")
        output_lines.append("")
        output_lines.append(scope_meta["description"])
        output_lines.append("")
        output_lines.append("## Pre-selection filters")
        output_lines.append("")
        output_lines.append(
            f"- Source MVEB(extended): **{len(source_task_names)}** tasks"
        )
        output_lines.append(
            f"- After scope filter (`{scope_key}`): **{len(scoped_tasks) + len(av_pref_dropped)}** "
            f"(-{len(scope_dropped)})"
        )
        if av_pref_dropped:
            output_lines.append(
                f"- After AV-variant preference: **{len(scoped_tasks)}** "
                f"(-{len(av_pref_dropped)})"
            )
        output_lines.append(
            f"- After annotation-provenance filter: **{len(annotation_valid)}** "
            f"(-{len(annotation_dropped)})"
        )
        output_lines.append(
            f"- After saturation/floor filter (best≤{args.sat_threshold}, "
            f"spread≥{args.floor_spread}, n≥{args.min_support}): "
            f"**{len(working_pool)}** (-{len(sat_dropped)})"
        )
        output_lines.append("")
        output_lines.append(
            f"- Must-include tasks in scope: **{len(must_keep)}** "
            "(bypass annotation and saturation filters)"
        )
        output_lines.append("")
        if av_pref_dropped:
            output_lines.append("### Dropped — non-AV variant (AV variant exists)")
            output_lines.append("")
            for t, reason in av_pref_dropped[:30]:
                output_lines.append(f"- `{t}` — {reason}")
            if len(av_pref_dropped) > 30:
                output_lines.append(f"- ... and {len(av_pref_dropped) - 30} more")
            output_lines.append("")
        if annotation_dropped:
            output_lines.append("### Dropped — annotation provenance")
            output_lines.append("")
            for t, reason in annotation_dropped[:30]:
                output_lines.append(f"- `{t}` — {reason}")
            if len(annotation_dropped) > 30:
                output_lines.append(f"- ... and {len(annotation_dropped) - 30} more")
            output_lines.append("")
        if sat_dropped:
            output_lines.append("### Dropped — saturated, floor, or low-support")
            output_lines.append("")
            for t, reason in sat_dropped[:30]:
                output_lines.append(f"- `{t}` — {reason}")
            if len(sat_dropped) > 30:
                output_lines.append(f"- ... and {len(sat_dropped) - 30} more")
            output_lines.append("")
        if must_keep:
            output_lines.append("### Must-include tasks (kept regardless)")
            output_lines.append("")
            for t in sorted(must_keep):
                output_lines.append(f"- `{t}`")
            output_lines.append("")
            missing = MUST_INCLUDE - set(scoped_tasks)
            if missing:
                output_lines.append("### Must-include tasks not in this scope (review)")
                output_lines.append("")
                for t in sorted(missing):
                    note = (
                        "" if t in source_task_names else "  *(not in MVEB(extended))*"
                    )
                    output_lines.append(f"- `{t}`{note}")
                output_lines.append("")
        output_lines.append("# MVEB Task Selection Analysis")
        output_lines.append("")
        output_lines.append("## Overview")
        output_lines.append(
            f"- **Source pool**: MVEB(extended) with {len(source_task_names)} tasks"
        )
        if excluded_count > 0:
            output_lines.append(
                f"- **Excluded tasks**: {excluded_count} ({', '.join(TASKS_TO_EXCLUDE)})"
            )
        output_lines.append(f"- **Working pool**: {len(filtered_source_tasks)} tasks")
        output_lines.append(
            f"- **Goal**: Select non-redundant tasks while preserving coverage"
        )
        output_lines.append("")

        output_lines.append("## Selection Rules")
        output_lines.append("")
        output_lines.append(
            "1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)"
        )
        output_lines.append(
            "2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task"
        )
        output_lines.append(
            "3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage"
        )
        output_lines.append("")

        # Get protected tasks (unique coverage) from filtered source pool
        unique_tasks = get_unique_coverage_tasks(filtered_source_tasks)
        protected = set()
        for dim, task_list in unique_tasks.items():
            for task, coverage in task_list:
                protected.add(task)

        # Add manually protected tasks
        for task in MANUALLY_PROTECTED_TASKS:
            if task in filtered_source_tasks:
                protected.add(task)

        # Must-include tasks bypass redundancy pruning too.
        protected |= must_keep

        output_lines.append(f"## Protected Tasks (Unique Coverage): {len(protected)}")
        output_lines.append("")
        for dim, task_list in unique_tasks.items():
            if task_list:
                output_lines.append(f"### {dim.replace('_', ' ').title()}")
                for task, coverage in task_list[:20]:  # Limit to first 20
                    output_lines.append(f"- {task} (unique: {coverage})")
                if len(task_list) > 20:
                    output_lines.append(f"- ... and {len(task_list) - 20} more")
                output_lines.append("")

        # Show manually protected tasks
        manual_in_source = [
            t for t in MANUALLY_PROTECTED_TASKS if t in source_task_names
        ]
        if manual_in_source:
            output_lines.append("### Manually Protected")
            for task in manual_in_source:
                output_lines.append(f"- {task}")
            output_lines.append("")

        # Task families analysis
        output_lines.append("## Task Families (Same Source Dataset)")
        output_lines.append("")
        families = identify_task_families(metadata_df)
        for family, tasks in sorted(families.items(), key=lambda x: -len(x[1])):
            if len(tasks) > 1:
                output_lines.append(f"### {family} ({len(tasks)} tasks)")
                for task in tasks:
                    task_meta = metadata_df[metadata_df["name"] == task].iloc[0]
                    output_lines.append(
                        f"- {task} ({task_meta['type']}, {task_meta['category']})"
                    )
                output_lines.append("")

        # Run selection at different thresholds
        results_by_threshold = {}
        for threshold in THRESHOLDS:
            remaining, removed = iterative_task_selection(
                results_df,
                filtered_source_tasks,
                metadata_df,
                threshold=threshold,
                protected_tasks=protected,
                prefer_remove_same_source=True,
            )
            # Post-processing: enforce retrieval direction preference (T2V over V2T)
            remaining, direction_removed = enforce_retrieval_direction_preference(
                remaining
            )
            removed = removed + direction_removed

            # Post-processing: deduplicate same-source families
            remaining, family_removed = deduplicate_same_source_families(
                remaining, results_df, metadata_df, protected_tasks=protected
            )
            removed = removed + family_removed
            results_by_threshold[threshold] = (remaining, removed)

        # Compute correlations for each threshold (compare against full source, not filtered)
        correlations_by_threshold = {}
        for threshold in THRESHOLDS:
            remaining, _ = results_by_threshold[threshold]
            spearman, pearson = compute_benchmark_correlation(
                results_df, source_task_names, remaining
            )
            correlations_by_threshold[threshold] = (spearman, pearson)

        # Load evaluation times for all models
        all_model_eval_times = {}
        source_eval_times = {}
        for short_name, model_name in EVAL_TIME_MODELS:
            model_times = load_eval_times(
                results_dir, model_name, filtered_source_tasks
            )
            if model_times:
                all_model_eval_times[short_name] = model_times
                total, count = compute_total_eval_time(
                    model_times, filtered_source_tasks
                )
                source_eval_times[short_name] = (total, count)

        if all_model_eval_times:
            output_lines.append("## Evaluation Time (MVEB Extended Working Pool)")
            output_lines.append("")
            output_lines.append("| Model | Time | Tasks w/ data |")
            output_lines.append("|-------|------|---------------|")
            for short_name, model_name in EVAL_TIME_MODELS:
                if short_name in source_eval_times:
                    total, count = source_eval_times[short_name]
                    output_lines.append(
                        f"| {short_name} ({model_name}) | {format_duration(total)} | {count}/{len(filtered_source_tasks)} |"
                    )
                else:
                    output_lines.append(
                        f"| {short_name} ({model_name}) | N/A | 0/{len(filtered_source_tasks)} |"
                    )
            output_lines.append("")
        else:
            output_lines.append("## Evaluation Time")
            output_lines.append("")
            output_lines.append("*No evaluation time data found*")
            output_lines.append("")

        # Task type short names
        type_short = {
            "Any2AnyRetrieval": "Retr",
            "VideoClassification": "Class",
            "VideoClustering": "Clust",
            "VideoMultilabelClassification": "MLC",
            "VideoPairClassification": "Pair",
            "VideoZeroshotClassification": "ZS",
            "VideoCentricQA": "QA",
        }

        # Summary table
        output_lines.append("## Selection Results Summary")
        output_lines.append("")
        model_short_names = [short for short, _ in EVAL_TIME_MODELS]
        if all_model_eval_times:
            model_headers = " | ".join(model_short_names)
            output_lines.append(
                f"| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | {model_headers} |"
            )
            model_sep = " | ".join(["---"] * len(model_short_names))
            output_lines.append(
                f"|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|{model_sep}|"
            )
        else:
            output_lines.append(
                "| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson |"
            )
            output_lines.append(
                "|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|"
            )

        original_coverage = get_coverage_analysis(filtered_source_tasks)

        # Compute eval times for each threshold (for all models)
        eval_times_by_threshold = {}
        for threshold in THRESHOLDS:
            remaining, _ = results_by_threshold[threshold]
            threshold_times = {}
            for short_name in model_short_names:
                if short_name in all_model_eval_times:
                    total_time, _ = compute_total_eval_time(
                        all_model_eval_times[short_name], remaining
                    )
                    threshold_times[short_name] = total_time
                else:
                    threshold_times[short_name] = None
            eval_times_by_threshold[threshold] = threshold_times

        for threshold in THRESHOLDS:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)
            spearman, pearson = correlations_by_threshold[threshold]
            threshold_times = eval_times_by_threshold[threshold]

            # Get type counts
            type_counts = coverage.get("type_counts", {})
            retr = type_counts.get("Any2AnyRetrieval", 0)
            cls = type_counts.get("VideoClassification", 0)
            clust = type_counts.get("VideoClustering", 0)
            mlc = type_counts.get("VideoMultilabelClassification", 0)
            pair = type_counts.get("VideoPairClassification", 0)
            zs = type_counts.get("VideoZeroshotClassification", 0)
            qa = type_counts.get("VideoCentricQA", 0)

            base_row = (
                f"| {threshold} | {len(remaining)} | {retr} | {cls} | {clust} | {mlc} | {pair} | {zs} | {qa} | "
                f"{coverage['n_languages']} | {coverage['n_domains']} | "
                f"{spearman:.4f} | {pearson:.4f}"
            )

            if all_model_eval_times:
                time_cols = []
                for short_name in model_short_names:
                    t = threshold_times.get(short_name)
                    if t is not None:
                        time_cols.append(format_duration(t))
                    else:
                        time_cols.append("N/A")
                output_lines.append(f"{base_row} | {' | '.join(time_cols)} |")
            else:
                output_lines.append(f"{base_row} |")

        output_lines.append("")
        output_lines.append(
            f"*Working pool: {len(filtered_source_tasks)} tasks, "
            f"{original_coverage['n_languages']} langs, "
            f"{original_coverage['n_domains']} doms*"
        )
        output_lines.append("")
        output_lines.append(
            "*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*"
        )
        output_lines.append("")

        # Detailed results for each threshold
        for threshold in THRESHOLDS:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)

            output_lines.append(f"## Threshold {threshold}")
            output_lines.append("")
            output_lines.append(
                f"**{len(source_task_names)} → {len(remaining)} tasks** ({len(removed)} removed)"
            )
            output_lines.append("")

            if remaining:
                output_lines.append("### Remaining Tasks")
                output_lines.append("")
                # Group remaining tasks by type for better organization
                remaining_tasks = mteb.get_tasks(tasks=remaining)
                by_type = defaultdict(list)
                for task in remaining_tasks:
                    by_type[task.metadata.type].append(task.metadata.name)

                for task_type, tasks in sorted(by_type.items()):
                    output_lines.append(f"#### {task_type} ({len(tasks)})")
                    for task_name in sorted(tasks):
                        task_obj = mteb.get_task(task_name)
                        meta = task_obj.metadata
                        domains = ", ".join(meta.domains) if meta.domains else "N/A"
                        output_lines.append(
                            f"- **{task_name}** - {meta.category}, {domains}"
                        )
                    output_lines.append("")

            # Coverage
            output_lines.append("### Coverage")
            output_lines.append(
                f"- Languages: {coverage['n_languages']} (was {original_coverage['n_languages']})"
            )
            output_lines.append(
                f"- Domains: {coverage['n_domains']} (was {original_coverage['n_domains']})"
            )
            output_lines.append(
                f"- Categories: {coverage['n_categories']} (was {original_coverage['n_categories']})"
            )
            output_lines.append(
                f"- Types: {coverage['n_types']} (was {original_coverage['n_types']})"
            )
            output_lines.append("")

        # Recommended task list (threshold 0.85)
        recommended_threshold = 0.85
        remaining, removed = results_by_threshold[recommended_threshold]

        output_lines.append(
            f"## Recommended MVEB Task List (threshold={recommended_threshold})"
        )
        output_lines.append("")
        output_lines.append(f"**Total: {len(remaining)} tasks**")
        output_lines.append("")

        # Group by type
        remaining_tasks = mteb.get_tasks(tasks=remaining)
        by_type = defaultdict(list)
        for task in remaining_tasks:
            by_type[task.metadata.type].append(task.metadata.name)

        for task_type, tasks in sorted(by_type.items()):
            output_lines.append(f"### {task_type} ({len(tasks)})")
            for t in sorted(tasks):
                task_obj = mteb.get_task(t)
                meta = task_obj.metadata
                langs = ", ".join(meta.languages[:3]) if meta.languages else "N/A"
                if meta.languages and len(meta.languages) > 3:
                    langs += f" (+{len(meta.languages) - 3} more)"
                domains = ", ".join(meta.domains) if meta.domains else "N/A"
                output_lines.append(f"- **{t}** - {meta.category}, {domains}")
            output_lines.append("")

        # Code block for benchmarks.py
        output_lines.append("### Code for benchmarks.py")
        output_lines.append("")
        output_lines.append("```python")
        output_lines.append("tasks=get_tasks(")
        output_lines.append("    tasks=[")
        for task_type, tasks in sorted(by_type.items()):
            output_lines.append(f"        # {task_type} ({len(tasks)})")
            for t in sorted(tasks):
                output_lines.append(f'        "{t}",')
        output_lines.append("    ]")
        output_lines.append("),")
        output_lines.append("```")
        output_lines.append("")

        # Write to markdown file
        output_path = Path(
            f"scripts/mveb_paper/mveb_task_selection_analysis.{scope_key}.md"
        )
        output_path.write_text("\n".join(output_lines))
        print(f"\nAnalysis written to: {output_path}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nSource: MVEB(extended) with {len(source_task_names)} tasks")
        if excluded_count > 0:
            print(f"Excluded: {excluded_count} tasks ({', '.join(TASKS_TO_EXCLUDE)})")
        print(f"Working pool: {len(filtered_source_tasks)} tasks")
        if source_eval_times:
            print("\nEval times for working pool:")
            for short_name, model_name in EVAL_TIME_MODELS:
                if short_name in source_eval_times:
                    total, count = source_eval_times[short_name]
                    print(
                        f"  {short_name}: {format_duration(total)} ({count}/{len(filtered_source_tasks)} tasks)"
                    )
        print(f"\nProtected tasks: {len(protected)}")
        print("\nResults by threshold:")
        # Simplified console output without per-model times
        print(
            f"  {'Thresh':<7} {'Tasks':<6} {'Retr':<5} {'Cls':<4} {'Clu':<4} {'MLC':<4} {'Pair':<5} {'ZS':<3} {'QA':<3} {'Spearman':<9} {'Pearson':<8}"
        )
        print(
            f"  {'-' * 7} {'-' * 6} {'-' * 5} {'-' * 4} {'-' * 4} {'-' * 4} {'-' * 5} {'-' * 3} {'-' * 3} {'-' * 9} {'-' * 8}"
        )
        for threshold in THRESHOLDS:
            remaining, removed = results_by_threshold[threshold]
            coverage = get_coverage_analysis(remaining)
            type_counts = coverage.get("type_counts", {})
            spearman, pearson = correlations_by_threshold[threshold]
            retr = type_counts.get("Any2AnyRetrieval", 0)
            cls = type_counts.get("VideoClassification", 0)
            clust = type_counts.get("VideoClustering", 0)
            mlc = type_counts.get("VideoMultilabelClassification", 0)
            pair = type_counts.get("VideoPairClassification", 0)
            zs = type_counts.get("VideoZeroshotClassification", 0)
            qa = type_counts.get("VideoCentricQA", 0)
            print(
                f"  {threshold:<7} {len(remaining):<6} {retr:<5} {cls:<4} {clust:<4} {mlc:<4} {pair:<5} {zs:<3} {qa:<3} {spearman:<9.4f} {pearson:<8.4f}"
            )

        print(
            f"\nRecommended (threshold={recommended_threshold}): {len(results_by_threshold[recommended_threshold][0])} tasks"
        )
        rec_spearman, rec_pearson = correlations_by_threshold[recommended_threshold]
        print(
            f"  Correlation with MVEB(extended): Spearman={rec_spearman:.4f}, Pearson={rec_pearson:.4f}"
        )
        print(f"\nFull analysis (with per-model eval times) saved to: {output_path}")


if __name__ == "__main__":
    main()
