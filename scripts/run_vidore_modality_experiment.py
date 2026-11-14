"""Compare document input modes on a complete ViDoRe v3.1 task subset.

Example (from the repository root, using its Nemotron runtime)::

    python scripts/run_vidore_modality_experiment.py \
        --output /tmp/mteb-vidore-modality-results

The default is the full HR corpus and all English (native) queries. This is
one task/subset check, not a ViDoRe leaderboard aggregate. No scores are uploaded.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mteb
from mteb.cache import ResultCache
from mteb.models import ModalityFilterWrapper

MODES = {"text": ["text"], "image": ["image"], "combined": ["image", "text"]}
MODEL_NAME = "nvidia/Nemotron-3.5-Embed-1B-BF16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Vidore3HrRetrieval.v2")
    parser.add_argument("--subsets", nargs="+", default=["english"])
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This BF16 experiment requires a CUDA GPU.")
    args.output.mkdir(parents=True, exist_ok=True)
    task = mteb.get_task(args.task, hf_subsets=args.subsets)
    task.load_data(num_proc=1)
    task.convert_v1_dataset_format_to_v2(num_proc=1)
    meta = mteb.get_model_meta(MODEL_NAME)
    # Persist runtime choices in loader_kwargs so the model can be restored;
    # experiment_kwargs is reserved here for the input-mode ablation.
    loader_kwargs = dict(meta.loader_kwargs)
    loader_kwargs["model_kwargs"] = {
        **loader_kwargs.get("model_kwargs", {}),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }
    meta = meta.model_copy(update={"loader_kwargs": loader_kwargs}, deep=True)
    counts = {
        subset: {
            split: {
                "corpus": len(data["corpus"]),
                "queries": len(data["queries"]),
                "qrel_query_ids": len(data["relevant_docs"]),
            }
            for split, data in splits.items()
        }
        for subset, splits in task.dataset.items()
    }
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Real-model verification of modality experiments; not a leaderboard aggregate",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
        "model": meta.name,
        "model_revision": meta.revision,
        "task": task.metadata.name,
        "dataset": task.metadata.dataset,
        "hf_subsets": task.hf_subsets,
        "counts": counts,
        "corpus_downsampled": False,
        "queries_downsampled": False,
        "batch_size": args.batch_size,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "mteb",
                "torch",
                "transformers",
                "sentence-transformers",
                "datasets",
            ]
        },
        "loader_kwargs": loader_kwargs,
        "modes": {mode: MODES[mode] for mode in args.modes},
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    model = meta.load_model(device=args.device)
    cache = ResultCache(args.output)
    summary = []
    for mode in args.modes:
        selected = ModalityFilterWrapper(model, corpus_modalities=MODES[mode])
        print(
            f"Starting mode={mode}, experiment={selected.mteb_model_meta.experiment_name}",
            flush=True,
        )
        started = time.monotonic()
        result = mteb.evaluate(
            selected,
            task,
            cache=cache,
            encode_kwargs={"batch_size": args.batch_size, "show_progress_bar": True},
            overwrite_strategy="always" if args.overwrite else "only-missing",
            co2_tracker=False,
            num_proc=1,
        )
        scores = result.task_results[0]
        entry = {
            "mode": mode,
            "experiment_name": selected.mteb_model_meta.experiment_name,
            "experiment_kwargs": selected.mteb_model_meta.experiment_kwargs,
            "ndcg_at_10": scores.main_score,
            "elapsed_seconds": time.monotonic() - started,
            "scores": scores.scores,
        }
        summary.append(entry)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(entry, indent=2), flush=True)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("Saved results:", flush=True)
    for path in sorted(args.output.rglob("*.json")):
        print(path.relative_to(args.output), flush=True)


if __name__ == "__main__":
    main()
