"""Omni vs Specialist Analysis for MOEB issue #5361.

Quantifies the within-modality penalty paid by omni models compared to
modality specialists, as a function of number of modalities supported.

Usage:
    python scripts/analysis/omni_vs_specialist.py
    python scripts/analysis/omni_vs_specialist.py --output results/omni_penalty.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RESULTS_DIR = Path.home() / ".cache" / "mteb" / "remote" / "results"

# Modalities we analyse (single-modality task subsets)
FOCUS_MODALITIES = ["text", "image", "audio", "video"]


def _load_model_scores(model_dir: Path) -> dict[str, float]:
    """Return {task_name: main_score} for all JSON result files under model_dir."""
    scores: dict[str, float] = {}
    for json_file in model_dir.glob("**/*.json"):
        try:
            data = json.loads(json_file.read_text())
            task_name = data.get("task_name") or json_file.stem
            # main_score lives at top-level or nested under test split
            main = data.get("scores", {})
            # walk splits to find main_score
            found: list[float] = []
            for split_scores in main.values():
                for subset_scores in split_scores:
                    if "main_score" in subset_scores:
                        found.append(subset_scores["main_score"])
            if found:
                scores[task_name] = float(np.mean(found))
        except Exception:
            pass
    return scores


def main(output: str | None = None) -> None:
    import mteb

    # --- Build task → modality mapping ---
    logger.info("Loading task metadata …")
    all_tasks = mteb.get_tasks(exclude_superseded=True)
    task_modality: dict[str, list[str]] = {}
    for t in all_tasks:
        mods = sorted(set(t.metadata.modalities))
        task_modality[t.metadata.name] = mods

    # tasks that are PURELY one modality (no cross-modal retrieval mix)
    pure_tasks: dict[str, list[str]] = {
        name: mods for name, mods in task_modality.items() if len(mods) == 1
    }

    # --- Build model → modalities mapping from model registry ---
    logger.info("Loading model metadata …")
    model_metas = mteb.get_model_metas()
    model_modalities: dict[str, list[str]] = {}
    for m in model_metas:
        if m.name and m.modalities:
            model_modalities[m.name] = sorted(m.modalities)

    # --- Load scores for every model that has cached results ---
    logger.info(f"Scanning {RESULTS_DIR} for cached results …")
    rows: list[dict] = []

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        model_slug = model_dir.name  # e.g. "openai__clip-vit-base-patch32"
        # Convert slug back to HF name (__ → /)
        model_name = model_slug.replace("__", "/", 1)

        mods = model_modalities.get(model_name)
        if mods is None:
            continue  # model not in registry or no modality info

        scores = _load_model_scores(model_dir)
        if not scores:
            continue

        n_modalities = len(mods)

        for focal_mod in FOCUS_MODALITIES:
            if focal_mod not in mods:
                continue  # this model doesn't support this modality → skip

            # Tasks that test exactly this modality
            focal_tasks = [t for t, tm in pure_tasks.items() if tm == [focal_mod]]
            task_scores = [scores[t] for t in focal_tasks if t in scores]
            if not task_scores:
                continue

            avg_score = float(np.mean(task_scores))
            rows.append(
                {
                    "model": model_name,
                    "n_modalities": n_modalities,
                    "modalities": "|".join(mods),
                    "focal_modality": focal_mod,
                    "n_tasks": len(task_scores),
                    "avg_score": avg_score,
                    "is_specialist": n_modalities == 1,
                }
            )

    if not rows:
        logger.error("No data found — check RESULTS_DIR or model registry.")
        return

    # --- Compute penalty per modality ---
    logger.info(f"Collected {len(rows)} model×modality observations.")

    results: list[dict] = []
    for focal_mod in FOCUS_MODALITIES:
        mod_rows = [r for r in rows if r["focal_modality"] == focal_mod]
        if not mod_rows:
            continue

        specialist_scores = [r["avg_score"] for r in mod_rows if r["is_specialist"]]
        if not specialist_scores:
            logger.warning(f"No specialists found for modality={focal_mod}")
            continue

        specialist_scores = [s for s in specialist_scores if not np.isnan(s)]
        best_specialist = max(specialist_scores)
        avg_specialist = float(np.nanmean(specialist_scores))

        # Group omni models by n_modalities
        by_n: dict[int, list[float]] = defaultdict(list)
        for r in mod_rows:
            if not r["is_specialist"]:
                by_n[r["n_modalities"]].append(r["avg_score"])

        print(f"\n{'='*60}")
        print(f"Modality: {focal_mod.upper()}")
        print(f"  Best specialist:  {best_specialist:.4f}")
        print(f"  Mean specialist:  {avg_specialist:.4f}")
        print(f"  n specialists:    {len(specialist_scores)}")

        for n in sorted(by_n):
            omni_scores = by_n[n]
            best_omni = max(omni_scores)
            mean_omni = float(np.mean(omni_scores))
            penalty_vs_best = best_specialist - best_omni
            penalty_pct = penalty_vs_best / best_specialist * 100 if best_specialist > 0 else float("nan")
            print(
                f"  n_modalities={n}: best={best_omni:.4f}  mean={mean_omni:.4f} "
                f"  penalty={penalty_vs_best:+.4f} ({penalty_pct:+.1f}%)  n_models={len(omni_scores)}"
            )
            results.append(
                {
                    "focal_modality": focal_mod,
                    "n_modalities": n,
                    "best_omni": best_omni,
                    "mean_omni": mean_omni,
                    "best_specialist": best_specialist,
                    "mean_specialist": avg_specialist,
                    "penalty_abs": penalty_vs_best,
                    "penalty_pct": penalty_pct,
                    "n_models": len(omni_scores),
                }
            )

    # --- Save CSV ---
    if output:
        import csv

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved summary to {out_path}")

    # --- Also dump the per-model rows for deeper analysis ---
    rows_path = Path(output).with_suffix(".rows.json") if output else Path("omni_rows.json")
    rows_path.write_text(json.dumps(rows, indent=2))
    logger.info(f"Per-model rows saved to {rows_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Path for CSV summary output")
    args = parser.parse_args()
    main(output=args.output)
