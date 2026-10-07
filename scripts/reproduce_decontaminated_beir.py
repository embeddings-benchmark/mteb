"""Run native MTEB evaluations for the eight LightOn decontaminated BEIR tasks.

Example:
    python scripts/reproduce_decontaminated_beir.py --device cuda --precision float16

Uses the registered BGE v1.5 encoders (CLS pooling, 512-token limit, cosine
similarity, and the standard query instruction). Model and dataset revisions
come from MTEB's registry. The cache contains full task JSONs and model metadata.
Tensor output lets MTEB convert embeddings to float32 for CPU scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mteb

TASKS = [
    "SciFactDecontaminated",
    "NFCorpusDecontaminated",
    "SciDocsDecontaminated",
    "FiQADecontaminated",
    "ArguAnaDecontaminated",
    "TrecCOVIDDecontaminated",
    "QuoraRetrievalDecontaminated",
    "Touche2020Decontaminated",
]
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=["float32", "float16"], default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--cache", type=Path, default=Path("results/decontaminated-beir")
    )
    args = parser.parse_args()
    cache = mteb.ResultCache(cache_path=args.cache / args.precision)
    for model_name in args.models:
        model = mteb.get_model(model_name, device=args.device)
        if args.precision == "float16":
            model.model.half()
        for task_name in args.tasks:
            print(f"Evaluating {model_name} on {task_name}", flush=True)
            mteb.evaluate(
                model,
                mteb.get_task(task_name),
                cache=cache,
                encode_kwargs={
                    "batch_size": args.batch_size,
                    "convert_to_tensor": True,
                    "show_progress_bar": True,
                },
                co2_tracker=False,
            )
        del model


if __name__ == "__main__":
    main()
