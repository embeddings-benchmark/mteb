"""Re-key a completed CoREB cache to the public router revision.

Run this only after publishing ``submission/huggingface-model`` and replacing
``TODO_AFTER_HF_UPLOAD`` in the MTEB ModelMeta with the returned commit SHA.
The source cache is preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import mteb
from mteb.models.model_implementations.keonkim_coreb_router import (
    coreb_task_type_router,
)
from mteb.results import TaskResult

logger = logging.getLogger(__name__)

TASKS = {
    "CorebC2CReranking",
    "CorebC2CRetrieval",
    "CorebC2TReranking",
    "CorebC2TRetrieval",
    "CorebT2CReranking",
    "CorebT2CRetrieval",
}
SOURCE_MODEL_PATHS = (
    "omlabs__coreb-type-router-f2llmv2-330m-c2llm-7b",
    "keonkim__coreb-task-type-router-f2llmv2-330m-c2llm-7b",
)
SOURCE_REVISION = "router-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("revision", help="40-character Hugging Face commit SHA")
    parser.add_argument(
        "--prepare-submission",
        action="store_true",
        help="Run ResultCache.submit_results(create_pr=False) after validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate, re-key, and optionally prepare the result submission."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("revision must be a lowercase 40-character commit SHA")
    if coreb_task_type_router.revision != args.revision:
        raise ValueError(
            "Update the MTEB ModelMeta revision to the same commit SHA first"
        )

    source_roots = [
        args.cache_path / "results" / model_path / SOURCE_REVISION
        for model_path in SOURCE_MODEL_PATHS
    ]
    result_paths = sorted(
        path
        for source_root in source_roots
        for path in source_root.rglob("Coreb*.json")
    )
    source_directories = {path.parent for path in result_paths}
    if len(source_directories) != 1:
        raise FileNotFoundError(
            "expected one source result directory below "
            f"{', '.join(str(path) for path in source_roots)}; "
            f"found {len(source_directories)}"
        )
    source = source_directories.pop()
    destination = (
        args.cache_path
        / "results"
        / coreb_task_type_router.model_name_as_path()
        / args.revision
    )
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    found_tasks = {TaskResult.from_disk(path).task_name for path in result_paths}
    if found_tasks != TASKS:
        raise ValueError(
            f"expected six CoREB tasks; missing={TASKS - found_tasks}, "
            f"unexpected={found_tasks - TASKS}"
        )

    shutil.copytree(source, destination)
    model_meta_path = destination / "model_meta.json"
    model_meta_path.write_text(
        json.dumps(coreb_task_type_router.to_dict(), indent=4, default=str) + "\n"
    )

    logger.info("Validated and prepared six results at %s", destination)
    if args.prepare_submission:
        response = mteb.ResultCache(args.cache_path).submit_results(
            coreb_task_type_router,
            create_pr=False,
        )
        logger.info("Submission response:\n%s", json.dumps(dict(response), indent=2))


if __name__ == "__main__":
    main()
