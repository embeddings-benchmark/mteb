"""Reupload zero-shot classification tasks to the Hugging Face Hub in the retrieval format, to `mteb/{task name}`.
If that repository already exists, `{suffix}` is appended (`mteb/{task name}-retrieval` by default).

Every task that still inherits `AbsTaskZeroShotClassification` is loaded in its current (inputs + labels) format,
converted by `AbsTaskZeroShotClassification.load_data` and pushed as `corpus` / `queries` / `qrels` configs. After
pushing, the descriptive statistics of the task are recomputed (`--skip-stats` to disable) and the new revision of each
repository is stored in the `--output` file. Point the task metadata to the new
repository, update `dataset["revision"]` with it and make the task inherit from `AbsTaskRetrieval`.

By default nothing is pushed (dry run). Examples:

    python scripts/reupload_zeroshot_tasks.py                              # list tasks that would be pushed
    python scripts/reupload_zeroshot_tasks.py --push                      # push to mteb/{task name} (or mteb/{task name}-retrieval if it exists)
    python scripts/reupload_zeroshot_tasks.py --push --org my-test-org     # push to another organization
    python scripts/reupload_zeroshot_tasks.py --push --tasks FGVCAircraftZeroShotClassification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi

import mteb
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def target_repo(api: HfApi, task: mteb.AbsTask, org: str, suffix: str) -> str:
    """Repository the retrieval version of the task is pushed to.

    It is named after the task. If such a repository already exists (e.g. it holds the legacy format), the suffix is added.
    """
    repo = f"{org}/{task.metadata.name}"
    if api.repo_exists(repo, repo_type="dataset"):
        repo += suffix
    return repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--push", action="store_true", help="Actually push (default: dry run)"
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Task names (default: all not yet reuploaded)"
    )
    parser.add_argument("--org", default="mteb")
    parser.add_argument(
        "--suffix",
        default="-retrieval",
        help="Appended to the repository name only if a repository named after the task already exists",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Do not recompute descriptive statistics after pushing",
    )
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zeroshot_reupload.json"),
        help="File with the pushed repositories and revisions (tasks in it are skipped)",
    )
    args = parser.parse_args()

    tasks = mteb.get_tasks(
        tasks=args.tasks,
        exclude_superseded=False,
        exclude_private=False,
        exclude_beta=False,
    )
    tasks = [t for t in tasks if isinstance(t, AbsTaskZeroShotClassification)]
    logger.info(f"{len(tasks)} tasks use the deprecated zero-shot class")

    done: dict[str, dict[str, str]] = (
        json.loads(args.output.read_text()) if args.output.exists() else {}
    )
    api = HfApi()
    failed: list[str] = []
    stats_failed: list[str] = []
    for task in tasks:
        name = task.metadata.name
        if name in done:
            logger.info(f"{name}: already pushed to {done[name]['path']}, skipping")
            continue
        repo = target_repo(api, task, args.org, args.suffix)
        if not args.push:
            logger.info(f"[dry run] {name} -> {repo}")
            continue

        logger.info(f"{name} -> {repo}")
        try:
            task.load_data(num_proc=args.num_proc)
            task.push_dataset_to_hub(repo, args.num_proc, private=args.private)
            revision = api.dataset_info(repo).sha
        except Exception:
            logger.exception(f"{name}: failed")
            failed.append(name)
            continue

        done[name] = {"path": repo, "revision": revision}
        # saved after each task to resume after a crash
        args.output.write_text(json.dumps(done, indent=2))

        if not args.skip_stats:
            # the data is already loaded and converted, so the statistics are retrieval statistics
            try:
                task.calculate_descriptive_statistics(
                    overwrite_results=True, num_proc=args.num_proc
                )
            except Exception:
                logger.exception(f"{name}: pushed, but statistics failed")
                stats_failed.append(name)

    if args.push:
        print(f"Pushed {len(done)} tasks, revisions are in {args.output}")
        print(
            "Update `dataset['path']`/`dataset['revision']` and switch to `AbsTaskRetrieval` for:"
        )
        print("\n".join(f"  {name}: {info['revision']}" for name, info in done.items()))
        if failed:
            print(f"Failed: {failed}")
        if stats_failed:
            print(
                f"Statistics failed (rerun scripts/recompute_zeroshot_stats.py): {stats_failed}"
            )


if __name__ == "__main__":
    main()
