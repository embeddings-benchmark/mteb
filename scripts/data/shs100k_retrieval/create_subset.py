#!/usr/bin/env python3
"""Create a smaller SHS100K A2A retrieval subset and optionally push to HF Hub.

This script subsets an existing Any2Any retrieval dataset by selecting complete
musical-work cliques (all tracks sharing the same `set_id` prefix in ids like
`<set_id>__<ver_id>`). Keeping full cliques guarantees valid qrels.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo, dataset_info


def _work_id(corpus_id: str) -> str:
    return corpus_id.split("__", 1)[0]


def _choose_works_exact_or_best(
    work_to_ids: dict[str, list[str]], target_size: int
) -> tuple[set[str], int]:
    """Pick works whose total track count is exact target when possible.

    Uses DP subset-sum over work sizes. If exact target is unreachable, picks
    the largest reachable size <= target.
    """
    works = sorted(work_to_ids)
    sizes = [len(work_to_ids[w]) for w in works]
    n = len(works)

    dp = [[False] * (target_size + 1) for _ in range(n + 1)]
    dp[n][0] = True
    for i in range(n - 1, -1, -1):
        sz = sizes[i]
        for s in range(target_size + 1):
            keep = dp[i + 1][s]
            take = s >= sz and dp[i + 1][s - sz]
            dp[i][s] = keep or take

    chosen_total = target_size
    if not dp[0][chosen_total]:
        reachable = [s for s in range(target_size + 1) if dp[0][s]]
        chosen_total = max(reachable)

    selected: set[str] = set()
    rem = chosen_total
    for i, work in enumerate(works):
        sz = sizes[i]
        if rem >= sz and dp[i + 1][rem - sz]:
            selected.add(work)
            rem -= sz
    return selected, chosen_total


def _load_split(repo: str, config: str, revision: str) -> Dataset:
    return load_dataset(repo, config, split="test", revision=revision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default="Wissam42/SHS100K-A2A")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--target-repo", default="Wissam42/SHS100K-A2A-1k")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/shs100k_a2a_1k"),
        help="Where to save subset locally when not pushing.",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.target_size < 2:
        raise SystemExit("--target-size must be at least 2")

    corpus = _load_split(args.source_repo, "corpus", args.source_revision).cast_column(
        "audio", Audio(decode=False)
    )
    queries = _load_split(
        args.source_repo, "queries", args.source_revision
    ).cast_column("audio", Audio(decode=False))
    qrels = _load_split(args.source_repo, "qrels", args.source_revision)

    work_to_ids: dict[str, list[str]] = defaultdict(list)
    for cid in corpus["id"]:
        work_to_ids[_work_id(cid)].append(cid)

    selected_works, selected_total = _choose_works_exact_or_best(
        work_to_ids, args.target_size
    )
    selected_cids = {cid for work in selected_works for cid in work_to_ids[work]}
    selected_qids = {f"q-{cid}" for cid in selected_cids}

    corpus_subset = corpus.filter(lambda row: row["id"] in selected_cids)
    queries_subset = queries.filter(lambda row: row["id"] in selected_qids)
    qrels_subset = qrels.filter(
        lambda row: (
            row["query-id"] in selected_qids and row["corpus-id"] in selected_cids
        )
    )

    # Keep hub representation aligned with existing task datasets.
    corpus_subset = corpus_subset.cast_column("audio", Audio())
    queries_subset = queries_subset.cast_column("audio", Audio())

    print(
        f"Selected works={len(selected_works)} "
        f"corpus={len(corpus_subset)} queries={len(queries_subset)} "
        f"qrels={len(qrels_subset)} target={args.target_size}"
    )
    if selected_total != args.target_size:
        print(
            f"Exact target not reachable with complete cliques; using {selected_total}."
        )

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.target_repo, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus_subset}).push_to_hub(
            args.target_repo, "corpus", token=token
        )
        DatasetDict({"test": queries_subset}).push_to_hub(
            args.target_repo, "queries", token=token
        )
        DatasetDict({"test": qrels_subset}).push_to_hub(
            args.target_repo, "qrels", token=token
        )
        sha = dataset_info(args.target_repo, token=token).sha
        args.work_dir.mkdir(parents=True, exist_ok=True)
        rev_path = args.work_dir / "hub_revision.txt"
        rev_path.write_text(sha + "\n", encoding="utf-8")
        print(f"Pushed {args.target_repo} @ {sha}")
        print(f"Wrote {rev_path}")
    else:
        out = args.work_dir / "mteb_export"
        out.mkdir(parents=True, exist_ok=True)
        DatasetDict({"test": corpus_subset}).save_to_disk(out / "corpus")
        DatasetDict({"test": queries_subset}).save_to_disk(out / "queries")
        DatasetDict({"test": qrels_subset}).save_to_disk(out / "qrels")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
