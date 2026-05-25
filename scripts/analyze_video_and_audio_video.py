"""Compare video-only vs video+audio performance across all paired tasks.

Three analyses:
1. v vs va paired tasks (classification, clustering, zeroshot, QA, retrieval)
   for each paired dataset, computes per-model audio delta (va - v score)
2. AV-intentional vs incidental datasets -- are gains larger where audio matters?
3. Cross-modal retrieval (v2a, a2v) -- high score = audio signal is redundant
   with video (they share the same embedding space)

Usage:
    # Download remote results first (one-time, ~GB):
    python scripts/analyze_video_vs_audio_video.py --download

    # Then analyse (or point at a local results dir):
    python scripts/analyze_video_vs_audio_video.py
    python scripts/analyze_video_vs_audio_video.py --cache-path /path/to/results
    python scripts/analyze_video_vs_audio_video.py --models model1 model2
    python scripts/analyze_video_vs_audio_video.py --output deltas.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

import mteb

# datasets annotated with audio as primary/co-primary signal
AV_INTENTIONAL = {
    "mteb/AVE-Dataset",
    "mteb/VGGSound",
    "mteb/RAVDESS_AV",
    "mteb/MELD",
    "mteb/MUSIC-AVQA_cls-preprocessed",
    "mteb/AVMeme-Exam",
    "mteb/Human-Animal-Cartoon",
}

# v-only category (va counterpart)
V_TO_VA = {
    "v2c": "va2c",
    "v2t": "va2t",
    "vt2t": "vat2t",
}


def build_task_index(
    tasks: list,
) -> dict[tuple[str, str, str], object]:
    """Index tasks by (dataset_path, task_type, category)."""
    index = {}
    for t in tasks:
        path = t.metadata.dataset.get("path", "")
        task_type = t.metadata.type
        cat = t.metadata.category
        index[(path, task_type, cat)] = t
    return index


def find_pairs(index: dict) -> list[tuple]:
    """Return list of (v_task, va_task) for every matched pair."""
    pairs = []
    seen = set()
    for (path, task_type, cat), v_task in index.items():
        if cat not in V_TO_VA:
            continue
        va_cat = V_TO_VA[cat]
        va_task = index.get((path, task_type, va_cat))
        if va_task is None:
            continue
        key = (path, task_type, cat)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((v_task, va_task))
    return pairs


def load_scores(
    cache: mteb.ResultCache,
    tasks: list,
    models: list[str] | None,
) -> dict[str, dict[str, float]]:
    """Return {task_name: {model_name: score}}."""
    results = cache.load_results(
        models=models,
        tasks=[t.metadata.name for t in tasks],
        include_remote=True,
    )
    df = results.to_dataframe(aggregation_level="task", format="long")
    if df.empty:
        return {}

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        scores[row["task_name"]][row["model_name"]] = row["score"]
    return scores


def compute_deltas(
    pairs: list[tuple],
    scores: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build a long DataFrame of (dataset, task_type, model, v_score, va_score, delta)."""
    rows = []
    for v_task, va_task in pairs:
        v_name = v_task.metadata.name
        va_name = va_task.metadata.name
        dataset_path = v_task.metadata.dataset.get("path", "")
        dataset_short = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
        task_type = v_task.metadata.type
        av_intent = dataset_path in AV_INTENTIONAL

        v_scores = scores.get(v_name, {})
        va_scores = scores.get(va_name, {})
        common_models = set(v_scores) & set(va_scores)

        for model in common_models:
            rows.append(
                {
                    "dataset": dataset_short,
                    "dataset_path": dataset_path,
                    "task_type": task_type,
                    "v_task": v_name,
                    "va_task": va_name,
                    "model": model,
                    "v_score": v_scores[model],
                    "va_score": va_scores[model],
                    "delta": va_scores[model] - v_scores[model],
                    "av_intentional": av_intent,
                }
            )
    return pd.DataFrame(rows)


def load_cross_modal_scores(
    cache: mteb.ResultCache,
    all_tasks: list,
    models: list[str] | None,
) -> pd.DataFrame:
    """Load scores for v2a / a2v retrieval tasks."""
    cross_tasks = [
        t for t in all_tasks if t.metadata.category in ("v2a", "a2v")
    ]
    if not cross_tasks:
        return pd.DataFrame()
    scores = load_scores(cache, cross_tasks, models)
    rows = []
    for t in cross_tasks:
        name = t.metadata.name
        cat = t.metadata.category
        dataset_path = t.metadata.dataset.get("path", "")
        dataset_short = dataset_path.split("/")[-1] if "/" in dataset_path else dataset_path
        for model, score in scores.get(name, {}).items():
            rows.append(
                {
                    "dataset": dataset_short,
                    "task": name,
                    "direction": cat,
                    "model": model,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main(models: list[str] | None, output: str | None, cache: mteb.ResultCache | None = None) -> None:
    if cache is None:
        cache = mteb.ResultCache()

    print("Loading video tasks...")
    all_tasks = list(mteb.get_tasks(modalities=["video"], exclude_beta=False))

    index = build_task_index(all_tasks)
    pairs = find_pairs(index)
    print(f"Found {len(pairs)} paired task groups (v vs va)")

    all_paired_tasks = [t for pair in pairs for t in pair]
    scores = load_scores(cache, all_paired_tasks, models)

    df = compute_deltas(pairs, scores)

    if df.empty:
        print("\nNo overlapping results found between v and va tasks.")
        print("Run models on these tasks first, or check your cache path.")
        return

    print_section("Audio delta per dataset  (va_score − v_score)")
    summary = (
        df.groupby(["dataset", "task_type", "av_intentional"])["delta"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta", "std": "std_delta", "count": "n_models"})
        .reset_index()
        .sort_values("mean_delta", ascending=False)
    )
    summary["av_intentional"] = summary["av_intentional"].map(
        {True: "yes", False: "no"}
    )
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:+.3f}".format)
    print(summary.to_string(index=False))

    print_section("Mean delta: AV-intentional datasets vs incidental")
    grouped = df.groupby("av_intentional")["delta"].agg(["mean", "std", "count"])
    grouped.index = grouped.index.map({True: "AV-intentional", False: "Incidental"})
    grouped.columns = ["mean_delta", "std_delta", "n_data_points"]
    grouped.index.name = None
    print(grouped.to_string())

    print_section("Per-model mean audio delta (across all paired datasets)")
    model_summary = (
        df.groupby("model")["delta"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta", "std": "std_delta", "count": "n_datasets"})
        .sort_values("mean_delta", ascending=False)
    )
    print(model_summary.to_string())

    print_section("Cross-modal retrieval scores  (v2a / a2v)")
    print("High score ≈ audio and video embeddings are well-aligned → audio may be redundant")
    cross_df = load_cross_modal_scores(cache, all_tasks, models)
    if cross_df.empty:
        print("No cross-modal retrieval results found.")
    else:
        cross_summary = (
            cross_df.groupby(["dataset", "direction"])["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["dataset", "direction"])
        )
        print(cross_summary.to_string(index=False))

    if output:
        df.to_csv(output, index=False)
        print(f"\nFull delta table saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to include (default: all in cache)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save full delta table to this CSV path",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Path to a local results directory (default: ~/.cache/mteb)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download latest results from the remote repo before analysing",
    )
    args = parser.parse_args()

    cache = mteb.ResultCache(cache_path=args.cache_path)
    if args.download:
        print("Downloading remote results (this may take a while)...")
        cache.download_from_remote()

    main(models=args.models, output=args.output, cache=cache)
