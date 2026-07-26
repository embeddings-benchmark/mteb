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
RETRIEVAL_TASKS = {task for task in TASKS if task.endswith("Retrieval")}
RERANKING_TASKS = TASKS - RETRIEVAL_TASKS
C2LLM_NAME = "codefuse-ai/C2LLM-7B"
C2LLM_REVISION = "c1dc16d6d64eb962c783bfb36a6d9c2f24a86dca"
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
        "--reranking-source",
        type=Path,
        required=True,
        help="Official C2LLM-7B result directory from embeddings-benchmark/results.",
    )
    parser.add_argument(
        "--prepare-submission",
        action="store_true",
        help="Run ResultCache.submit_results(create_pr=False) after validation.",
    )
    return parser.parse_args()


def _validate_reranking_source(source: Path) -> list[Path]:
    meta_path = source / "model_meta.json"
    meta = json.loads(meta_path.read_text())
    if (meta.get("name"), meta.get("revision")) != (C2LLM_NAME, C2LLM_REVISION):
        raise ValueError(
            f"unexpected C2 source metadata in {meta_path}: "
            f"{meta.get('name')} at {meta.get('revision')}"
        )
    result_paths = [source / f"{task}.json" for task in RERANKING_TASKS]
    tasks = {TaskResult.from_disk(path).task_name for path in result_paths}
    if tasks != RERANKING_TASKS:
        raise ValueError("official C2 source does not contain all reranking tasks")
    return result_paths


def _retain_retrieval_run_settings(result_directory: Path) -> None:
    path = result_directory / "run_settings.jsonl"
    if not path.exists():
        return

    retrieval_settings = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("task") in RETRIEVAL_TASKS:
            retrieval_settings.append(entry)
    path.write_text(
        "".join(json.dumps(entry, default=str) + "\n" for entry in retrieval_settings)
    )


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
    missing_tasks = RETRIEVAL_TASKS - found_tasks
    unexpected_tasks = found_tasks - TASKS
    if missing_tasks or unexpected_tasks:
        raise ValueError(
            "source must contain all CoREB Retrieval tasks; "
            f"missing={missing_tasks}, unexpected={unexpected_tasks}"
        )

    c2_result_paths = _validate_reranking_source(args.reranking_source)

    shutil.copytree(source, destination)
    for result_path in c2_result_paths:
        shutil.copy2(result_path, destination / result_path.name)

    _retain_retrieval_run_settings(destination)
    prepared_tasks = {
        TaskResult.from_disk(path).task_name for path in destination.glob("Coreb*.json")
    }
    if prepared_tasks != TASKS:
        raise ValueError(
            f"expected six prepared CoREB tasks; missing={TASKS - prepared_tasks}, "
            f"unexpected={prepared_tasks - TASKS}"
        )

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
