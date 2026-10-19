"""Recompute descriptive statistics of zero-shot classification tasks (now retrieval-shaped statistics).

Each task is loaded, its statistics are calculated with `calculate_descriptive_statistics(overwrite_results=True)` and
written to `mteb/descriptive_stats/<task type>/<task name>.json`. Tasks still using the deprecated
`AbsTaskZeroShotClassification` (not reuploaded yet) are skipped. Names of recomputed tasks are saved to `--output`, so
the script can be rerun and only processes new tasks (tasks whose statistics file already has retrieval statistics count as done; use `--force` to recompute all). Tasks that fail are reported at the end and can be retried.

Examples:

    python scripts/recompute_zeroshot_stats.py                                  # all zero-shot tasks
    python scripts/recompute_zeroshot_stats.py --tasks CIFAR10ZeroShot RenderedSST2
    python scripts/recompute_zeroshot_stats.py --num-proc 4                     # rerun: only tasks not in the output file
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

import mteb
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ZEROSHOT_TYPES = {
    "ZeroShotClassification",
    "AudioZeroshotClassification",
    "VideoZeroshotClassification",
}


def has_new_stats(task: mteb.AbsTask) -> bool:
    """Old zero-shot statistics have no retrieval keys (`num_queries`, `num_documents`)."""
    path = task.metadata.descriptive_stat_path
    return path.exists() and "num_queries" in path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tasks", nargs="+", help="Task names (default: all zero-shot tasks)"
    )
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zeroshot_stats_done.json"),
        help="File with the names of tasks whose statistics are already recomputed (they are skipped)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even tasks that are in the output file or already have retrieval statistics",
    )
    args = parser.parse_args()

    done: list[str] = (
        json.loads(args.output.read_text()) if args.output.exists() else []
    )

    tasks = mteb.get_tasks(
        tasks=args.tasks,
        exclude_superseded=False,
        exclude_private=False,
        exclude_beta=False,
    )
    # tasks that are not reuploaded yet still inherit the deprecated class
    tasks = [
        t
        for t in tasks
        if t.metadata.type in ZEROSHOT_TYPES
        and not isinstance(t, AbsTaskZeroShotClassification)
    ]
    logger.info(f"{len(tasks)} reuploaded zero-shot tasks, {len(done)} already done")

    failed: list[str] = []
    for task in tqdm(tasks, total=len(tasks), desc="recomputing statistics"):
        name = task.metadata.name
        if not args.force and (name in done or has_new_stats(task)):
            logger.info(f"{name}: already recomputed, skipping")
            if name not in done:
                done.append(name)
            continue
        logger.info(f"{name}: recomputing")
        try:
            task.calculate_descriptive_statistics(
                overwrite_results=True, num_proc=args.num_proc
            )
        except Exception:
            logger.exception(f"{name}: failed")
            failed.append(name)
            continue
        done.append(name)
        # saved after each task to resume after a crash
        args.output.write_text(json.dumps(done, indent=2))

    print(f"Recomputed {len(done)} tasks, the list is in {args.output}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
