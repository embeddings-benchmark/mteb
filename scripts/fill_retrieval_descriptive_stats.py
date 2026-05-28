#!/usr/bin/env python3
"""
Fetches num_queries and num_documents from the HuggingFace Datasets Server API
and updates descriptive statistics JSON files for retrieval-type tasks.

Tasks whose HuggingFace datasets use a non-standard config layout (e.g. only a
`default` config with no separate `corpus` / `queries` configs) cannot be filled
automatically and are written to `scripts/tasks_needing_manual_stats.json`.

Usage:
    python scripts/fill_retrieval_descriptive_stats.py
    python scripts/fill_retrieval_descriptive_stats.py --token hf_xxx
    python scripts/fill_retrieval_descriptive_stats.py --dry-run
    python scripts/fill_retrieval_descriptive_stats.py --task-types Retrieval Reranking
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

import mteb

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"

TARGET_TASK_TYPES = [
    "Retrieval",
    "Reranking",
    "Any2AnyRetrieval",
    "Any2AnyMultilingualRetrieval",
    "InstructionRetrieval",
    "InstructionReranking",
]

AGGREGATE_PATH = "aggregate tasks do not have a path"

UNFILLABLE_JSON = Path(__file__).parent / "tasks_needing_manual_stats.json"


# ---------------------------------------------------------------------------
# HuggingFace Datasets Server helpers
# ---------------------------------------------------------------------------


def fetch_dataset_size(repo_id: str, token: str | None = None) -> dict | None:
    """Fetch /size for all configs and splits of a HuggingFace dataset."""
    url = f"{DATASETS_SERVER_URL}/size?dataset={repo_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code in (401, 403, 404):
            logger.warning(
                "  %s – HTTP %d from Datasets Server", repo_id, resp.status_code
            )
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("  Failed to fetch size for %s: %s", repo_id, exc)
        return None


def find_split_rows(
    splits_info: list[dict],
    config: str,
    preferred_splits: list[str],
) -> int | None:
    """Return num_rows for *config*, trying each candidate split name in order.

    Falls back to the single split when there is exactly one split for the config.
    """
    matching = [s for s in splits_info if s["config"] == config]
    if not matching:
        return None
    for split_name in preferred_splits:
        for entry in matching:
            if entry["split"] == split_name:
                return entry["num_rows"]
    if len(matching) == 1:
        return matching[0]["num_rows"]
    return None


def get_corpus_rows(
    splits_info: list[dict], subset: str | None, eval_split: str
) -> int | None:
    config = f"{subset}-corpus" if subset else "corpus"
    return find_split_rows(splits_info, config, [eval_split, "corpus", "train"])


def get_queries_rows(
    splits_info: list[dict], subset: str | None, eval_split: str
) -> int | None:
    for suffix in ("queries", "query"):
        config = f"{subset}-{suffix}" if subset else suffix
        rows = find_split_rows(
            splits_info, config, [eval_split, "queries", "query", "train"]
        )
        if rows is not None:
            return rows
    return None


def has_corpus_config(splits_info: list[dict], subset: str | None) -> bool:
    config = f"{subset}-corpus" if subset else "corpus"
    return any(s["config"] == config for s in splits_info)


# ---------------------------------------------------------------------------
# Stats JSON helpers
# ---------------------------------------------------------------------------


def update_entry(entry: dict, num_documents: int, num_queries: int) -> bool:
    """Inject num_documents / num_queries into a stats dict. Returns True if changed."""
    if "num_queries" in entry and "num_documents" in entry:
        return False
    entry["num_documents"] = num_documents
    entry["num_queries"] = num_queries
    return True


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------


def process_task(
    task: mteb.AbsTask,
    size_cache: dict[str, dict | None],
    token: str | None,
    dry_run: bool,
) -> str:
    """Process one task.

    Returns one of:
      "updated"    – stats file was changed
      "unchanged"  – fields already present; nothing to do
      "skipped"    – aggregate task or no HF path
      "failed"     – HF fetch failed
      "unfillable" – non-standard config layout; needs manual work
    """
    meta = task.metadata
    repo_id = meta.dataset.get("path", "")

    if not repo_id or repo_id == AGGREGATE_PATH:
        logger.debug("Skipping aggregate task: %s", meta.name)
        return "skipped"

    if repo_id not in size_cache:
        size_cache[repo_id] = fetch_dataset_size(repo_id, token=token)
        time.sleep(0.2)

    size_data = size_cache[repo_id]
    if size_data is None:
        return "failed"

    splits_info: list[dict] = size_data.get("size", {}).get("splits", [])
    if not splits_info:
        logger.warning("  No splits in response for %s", repo_id)
        return "failed"

    # Quick check: does this dataset expose any corpus config at all?
    is_multilingual = meta.is_multilingual
    first_subset = meta.hf_subsets[0] if is_multilingual else None
    if not has_corpus_config(splits_info, first_subset):
        logger.info("  Non-standard config for %s – flagging as unfillable", meta.name)
        return "unfillable"

    # Load or create the stats dict
    stats_path: Path = meta.descriptive_stat_path
    if stats_path.exists():
        with stats_path.open() as fh:
            stats: dict = json.load(fh)
    else:
        stats = {}

    changed = False

    for eval_split in meta.eval_splits:
        if eval_split not in stats:
            stats[eval_split] = {}
        split_stats = stats[eval_split]

        if is_multilingual:
            total_docs = 0
            total_queries = 0
            any_subset_found = False
            subset_descriptive = split_stats.get("hf_subset_descriptive_stats", {})

            for hf_subset in meta.hf_subsets:
                if not has_corpus_config(splits_info, hf_subset):
                    logger.debug("    No corpus config for subset=%s", hf_subset)
                    continue

                docs = get_corpus_rows(splits_info, hf_subset, eval_split)
                if docs is None:
                    logger.debug(
                        "    No corpus rows for subset=%s split=%s",
                        hf_subset,
                        eval_split,
                    )
                    continue

                # Prefer deriving queries from existing num_samples for consistency
                queries: int | None = None
                if hf_subset in subset_descriptive:
                    num_samples = subset_descriptive[hf_subset].get("num_samples")
                    if num_samples is not None:
                        queries = num_samples - docs

                if queries is None:
                    queries = get_queries_rows(splits_info, hf_subset, eval_split)

                if queries is None:
                    logger.debug(
                        "    No query rows for subset=%s split=%s",
                        hf_subset,
                        eval_split,
                    )
                    continue

                if queries < 0:
                    logger.warning(
                        "    Negative num_queries for subset=%s (docs=%d, num_samples=%s)",
                        hf_subset,
                        docs,
                        subset_descriptive.get(hf_subset, {}).get("num_samples"),
                    )
                    continue

                any_subset_found = True
                total_docs += docs
                total_queries += queries

                if hf_subset in subset_descriptive:
                    changed |= update_entry(
                        subset_descriptive[hf_subset], docs, queries
                    )

            if any_subset_found:
                changed |= update_entry(split_stats, total_docs, total_queries)
                if "num_samples" not in split_stats:
                    split_stats["num_samples"] = total_docs + total_queries
                    changed = True

        else:
            docs = get_corpus_rows(splits_info, None, eval_split)
            if docs is None:
                logger.warning(
                    "  No corpus rows for %s split=%s", meta.name, eval_split
                )
                continue

            num_samples = split_stats.get("num_samples")
            if num_samples is not None:
                queries = num_samples - docs
            else:
                queries = get_queries_rows(splits_info, None, eval_split)

            if queries is None:
                logger.warning(
                    "  Cannot determine num_queries for %s split=%s",
                    meta.name,
                    eval_split,
                )
                continue

            if queries < 0:
                logger.warning(
                    "  Negative num_queries for %s split=%s (docs=%d, num_samples=%d)",
                    meta.name,
                    eval_split,
                    docs,
                    num_samples or -1,
                )
                continue

            if split_stats:
                changed |= update_entry(split_stats, docs, queries)
            else:
                stats[eval_split] = {
                    "num_samples": docs + queries,
                    "num_documents": docs,
                    "num_queries": queries,
                }
                changed = True

    if not changed:
        return "unchanged"

    if not dry_run:
        with stats_path.open("w") as fh:
            json.dump(stats, fh, indent=4, ensure_ascii=False)
        logger.info("  Wrote: %s", stats_path.name)

    return "updated"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=TARGET_TASK_TYPES,
        metavar="TYPE",
        help="Task types to process (default: all retrieval-type tasks)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token for private / gated datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any files; just report what would change",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tasks = mteb.get_tasks(task_types=args.task_types)
    logger.info("Loaded %d tasks of types: %s", len(tasks), args.task_types)

    size_cache: dict[str, dict | None] = {}
    counts: dict[str, int] = {
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "failed": 0,
        "unfillable": 0,
    }
    unfillable: list[dict] = []

    for task in tqdm(tasks, desc="Filling stats", unit="task"):
        tqdm.write(f"Processing {task.metadata.name} ({task.metadata.type})")
        try:
            result = process_task(
                task, size_cache, token=args.token, dry_run=args.dry_run
            )
        except Exception as exc:
            logger.error("  Error processing %s: %s", task.metadata.name, exc)
            result = "failed"

        counts[result] += 1

        if result == "unfillable":
            unfillable.append(
                {
                    "name": task.metadata.name,
                    "type": task.metadata.type,
                    "dataset": task.metadata.dataset,
                    "eval_splits": task.metadata.eval_splits,
                    "is_multilingual": task.metadata.is_multilingual,
                }
            )

    # Write unfillable tasks for manual follow-up
    if unfillable and not args.dry_run:
        existing: list[dict] = []
        if UNFILLABLE_JSON.exists():
            with UNFILLABLE_JSON.open() as fh:
                existing = json.load(fh)
        existing_names = {t["name"] for t in existing}
        new_entries = [t for t in unfillable if t["name"] not in existing_names]
        if new_entries:
            existing.extend(new_entries)
            with UNFILLABLE_JSON.open("w") as fh:
                json.dump(existing, fh, indent=2)
            logger.info(
                "Wrote %d unfillable tasks to %s", len(new_entries), UNFILLABLE_JSON
            )

    logger.info(
        "Done. updated=%d  unchanged=%d  skipped=%d  failed=%d  unfillable=%d%s",
        counts["updated"],
        counts["unchanged"],
        counts["skipped"],
        counts["failed"],
        counts["unfillable"],
        "  (dry-run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
