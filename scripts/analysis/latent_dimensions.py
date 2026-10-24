#!/usr/bin/env python3
"""PCA analysis of latent performance dimensions across MTEB models.

Loads cached MTEB results (~/.cache/mteb/remote/results/), builds a model×task
score matrix, imputes missing values, and runs PCA to reveal the underlying
performance structure.  Each principal component is labelled by the modality
of the tasks that load most strongly onto it.

Addresses MTEB issue #5367.

Examples:
  uv run python scripts/analysis/latent_dimensions.py \\
      --out results/latent_dimensions.json \\
      --min-model-coverage 50 \\
      --n-components 10
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

_RESULTS_DIR = Path.home() / ".cache" / "mteb" / "remote" / "results"

# Keyword patterns for fallback modality inference when mteb registry lacks the task
_MODALITY_PATTERNS: list[tuple[list[str], str]] = [
    (["Audio", "audio", "Speech", "speech", "Spoken", "spoken", "Sound"], "audio"),
    (["Video", "video"], "video"),
    (
        [
            "I2T",
            "T2I",
            "Image",
            "image",
            "Flickr",
            "COCO",
            "Visual",
            "visual",
            "BLINK",
            "CIRR",
            "EDIS",
            "ARO",
            "Fashion",
            "Hateful",
            "SugarCrepe",
            "WinoGround",
            "VQA",
            "MMStar",
            "Winoground",
        ],
        "image",
    ),
]

_OMNI_KEYWORDS = [
    "omni",
    "Omni",
    "clip",
    "CLIP",
    "siglip",
    "SigLIP",
    "siglip",
    "Voyage",
    "voyage",
    "BLIP",
    "blip",
    "LaCLIP",
    "LCO",
    "BidirLM",
    "OmniEmbed",
    "ebind",
    "Haon",
    "Tevatron",
]


def _infer_modality_from_name(task_fname: str) -> str:
    name = task_fname.replace(".json", "")
    for keywords, modality in _MODALITY_PATTERNS:
        if any(kw in name for kw in keywords):
            return modality
    return "text"


def _build_task_modality_map() -> dict[str, str]:
    """Map task filename -> primary modality using mteb registry + fallback."""
    import mteb

    task_modalities: dict[str, str] = {}
    try:
        all_tasks = mteb.get_tasks()
        for t in all_tasks:
            fname = t.metadata.name + ".json"
            mods = t.metadata.modalities
            # Use first modality if multiple (e.g. ['text','image'] → 'cross-modal')
            if len(mods) == 1:
                task_modalities[fname] = mods[0]
            elif len(mods) > 1:
                task_modalities[fname] = "cross-modal"
    except Exception:
        pass
    return task_modalities


def _model_type(model_dir: str, task_fnames: set[str]) -> str:
    has_img = any(
        "I2T" in t or "T2I" in t or "Flickr" in t or "MSCOCOI" in t for t in task_fnames
    )
    has_aud = any("Audio" in t or "Spoken" in t or "Speech" in t for t in task_fnames)
    if has_img and has_aud:
        return "omni"
    if has_img:
        return "image"
    if has_aud:
        return "audio"
    return "text"


def _load_results(
    results_dir: Path,
    min_model_coverage: int,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str], dict[str, str]]:
    """Return (matrix, model_names, task_names, task_modality, model_type)."""
    # First pass: collect scores per model
    raw: dict[str, dict[str, float]] = {}
    model_task_sets: dict[str, set[str]] = {}

    for model_dir in sorted(os.listdir(results_dir)):
        model_path = results_dir / model_dir
        if not model_path.is_dir():
            continue
        revs = [r for r in os.listdir(model_path) if (model_path / r).is_dir()]
        if not revs:
            continue
        rev_path = model_path / revs[0]

        scores: dict[str, float] = {}
        task_fnames: set[str] = set()
        for fname in os.listdir(rev_path):
            if not fname.endswith(".json") or fname == "model_meta.json":
                continue
            task_fnames.add(fname)
            try:
                with open(rev_path / fname) as f:
                    data = json.load(f)
                # Extract main_score from test split
                split_scores = data.get("scores", {})
                test_scores = split_scores.get(
                    "test", split_scores.get("validation", [])
                )
                if test_scores:
                    ms = test_scores[0].get("main_score")
                    if ms is not None:
                        scores[fname] = float(ms)
            except Exception:
                continue

        if scores:
            # Use human-readable model name (replace __ with /)
            model_name = model_dir.replace("__", "/")
            raw[model_name] = scores
            model_task_sets[model_name] = task_fnames

    # Count how many models have each task
    task_counts: dict[str, int] = defaultdict(int)
    for scores in raw.values():
        for t in scores:
            task_counts[t] += 1

    # Keep only tasks with sufficient model coverage
    kept_tasks = sorted(t for t, n in task_counts.items() if n >= min_model_coverage)

    model_names = sorted(raw.keys())
    n_models = len(model_names)
    n_tasks = len(kept_tasks)

    print(f"Models: {n_models}  Tasks (≥{min_model_coverage} models): {n_tasks}")

    # Build matrix with NaN for missing
    matrix = np.full((n_models, n_tasks), np.nan)
    task_idx = {t: i for i, t in enumerate(kept_tasks)}
    for mi, model in enumerate(model_names):
        for task, score in raw[model].items():
            if task in task_idx:
                matrix[mi, task_idx[task]] = score

    # Task modality map
    registry_mods = _build_task_modality_map()
    task_modality: dict[str, str] = {}
    for t in kept_tasks:
        if t in registry_mods:
            task_modality[t] = registry_mods[t]
        else:
            task_modality[t] = _infer_modality_from_name(t)

    # Model type
    mtype: dict[str, str] = {}
    for model in model_names:
        mtype[model] = _model_type(model, model_task_sets.get(model, set()))

    return matrix, model_names, kept_tasks, task_modality, mtype


def _impute_and_scale(matrix: np.ndarray) -> np.ndarray:
    """Column-mean imputation then z-score standardisation."""
    col_means = np.nanmean(matrix, axis=0)
    mat = matrix.copy()
    for j in range(mat.shape[1]):
        nans = np.isnan(mat[:, j])
        mat[nans, j] = col_means[j]

    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    mat = (mat - mat.mean(axis=0)) / col_std
    return mat


def run_pca(
    matrix: np.ndarray,
    model_names: list[str],
    task_names: list[str],
    task_modality: dict[str, str],
    model_type: dict[str, str],
    n_components: int,
) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    mat_scaled = _impute_and_scale(matrix)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(mat_scaled)  # (n_models, n_components)
    loadings = pca.components_  # (n_components, n_tasks)

    var_explained = pca.explained_variance_ratio_.tolist()

    # Per-component analysis
    components_out: list[dict] = []
    for k in range(n_components):
        load_k = loadings[k]  # (n_tasks,)

        # Top-10 positive and negative tasks
        top_pos_idx = np.argsort(-load_k)[:10]
        top_neg_idx = np.argsort(load_k)[:10]

        top_pos = [
            {
                "task": task_names[i],
                "modality": task_modality[task_names[i]],
                "loading": round(float(load_k[i]), 4),
            }
            for i in top_pos_idx
        ]
        top_neg = [
            {
                "task": task_names[i],
                "modality": task_modality[task_names[i]],
                "loading": round(float(load_k[i]), 4),
            }
            for i in top_neg_idx
        ]

        # Average loading by modality
        mod_loads: dict[str, list[float]] = defaultdict(list)
        for j, t in enumerate(task_names):
            mod_loads[task_modality[t]].append(float(load_k[j]))
        avg_by_modality = {
            mod: round(float(np.mean(vals)), 4)
            for mod, vals in sorted(mod_loads.items())
        }

        components_out.append(
            {
                "pc": k + 1,
                "variance_explained": round(var_explained[k], 4),
                "cumulative_variance": round(float(sum(var_explained[: k + 1])), 4),
                "avg_loading_by_modality": avg_by_modality,
                "top_positive_tasks": top_pos,
                "top_negative_tasks": top_neg,
            }
        )

    # Model coordinates
    model_coords: list[dict] = []
    for i, model in enumerate(model_names):
        model_coords.append(
            {
                "model": model,
                "type": model_type.get(model, "text"),
                "pc_coords": {
                    f"pc{k + 1}": round(float(coords[i, k]), 4)
                    for k in range(n_components)
                },
            }
        )

    # Task count by modality
    mod_counts: dict[str, int] = defaultdict(int)
    for t in task_names:
        mod_counts[task_modality[t]] += 1

    # Model count by type
    type_counts: dict[str, int] = defaultdict(int)
    for m in model_names:
        type_counts[model_type.get(m, "text")] += 1

    return {
        "summary": {
            "n_models": len(model_names),
            "n_tasks": len(task_names),
            "tasks_by_modality": dict(sorted(mod_counts.items())),
            "models_by_type": dict(sorted(type_counts.items())),
            "variance_explained": [round(v, 4) for v in var_explained],
            "cumulative_variance_top5": round(float(sum(var_explained[:5])), 4),
        },
        "components": components_out,
        "model_coordinates": model_coords,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR))
    parser.add_argument(
        "--min-model-coverage",
        type=int,
        default=50,
        help="Minimum number of models that must have evaluated a task (default 50)",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading results from {results_dir}...")

    matrix, model_names, task_names, task_modality, mtype = _load_results(
        results_dir, args.min_model_coverage
    )

    # Print modality breakdown of tasks
    from collections import Counter

    mod_ctr = Counter(task_modality[t] for t in task_names)
    print("Task modalities:", dict(sorted(mod_ctr.items())))

    type_ctr = Counter(mtype.get(m, "text") for m in model_names)
    print("Model types:", dict(sorted(type_ctr.items())))

    print(f"\nRunning PCA ({args.n_components} components)...")
    result = run_pca(
        matrix, model_names, task_names, task_modality, mtype, args.n_components
    )

    # Print summary to console
    print("\n=== Variance explained ===")
    for c in result["components"]:
        print(
            f"  PC{c['pc']:2d}  {c['variance_explained'] * 100:5.1f}%  "
            f"(cumulative {c['cumulative_variance'] * 100:5.1f}%)  "
            f"modality weights: {c['avg_loading_by_modality']}"
        )

    print("\n=== Top positive tasks per PC (first 5 PCs) ===")
    for c in result["components"][:5]:
        tasks_str = ", ".join(
            f"{e['task'].replace('.json', '').replace('Retrieval', 'Ret')} [{e['modality']}]"
            for e in c["top_positive_tasks"][:5]
        )
        print(f"  PC{c['pc']}: {tasks_str}")

    output = json.dumps(result, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"\nResults written to {out_path}")
    else:
        print("\n=== FULL OUTPUT ===")
        print(output)


if __name__ == "__main__":
    main()
