"""Create submission-ready MTEB result files for the ToolRetrieval tasks.

This example deliberately relies on the public MTEB model and evaluation APIs.
It does not reproduce the ToolRet paper's model-specific preprocessing; the
output is the normal MTEB evaluation of the two tasks and can be submitted to
the results repository without conversion.

Run from the repository root::

    python scripts/reproduce_toolret.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --device cuda

The default cache folder is the repository root, so MTEB writes the canonical
artifacts to ``results/<model>/<revision>/``. That directory contains one full
JSON result per task, ``model_meta.json``, and ``run_settings.jsonl``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mteb
from mteb.models import CachedEmbeddingWrapper


TASKS = ("ToolRetrieval", "ToolRetrievalInstruction")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a public embedding model on the ToolRetrieval tasks."
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model name accepted by mteb.get_model.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face model revision. Defaults to MTEB metadata.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to mteb.get_model (for example, cuda or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size passed to MTEB.",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path("."),
        help="Parent directory for MTEB's canonical results/ output directory.",
    )
    args = parser.parse_args()

    model = mteb.get_model(args.model, args.revision, device=args.device)
    tasks = mteb.get_tasks(tasks=TASKS)
    cache = mteb.ResultCache(args.cache_folder)
    cached_model = CachedEmbeddingWrapper(model, args.cache_folder / "embedding_cache")
    try:
        results = mteb.evaluate(
            cached_model,
            tasks,
            cache=cache,
            overwrite_strategy="always",
            encode_kwargs={"batch_size": args.batch_size},
            show_progress_bar=True,
        )
    finally:
        cached_model.close()

    for task_result in results.task_results:
        path = cache.get_task_result_path(
            task_result.task_name,
            results.model_name,
            results.model_revision,
        )
        print(path)


if __name__ == "__main__":
    main()
