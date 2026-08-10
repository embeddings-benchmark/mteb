"""One-off audit: do qrels ever reference query/corpus IDs that don't exist?

Not part of any PR. Checks a real, structural invariant that
`RetrievalDatasetLoader` (mteb/abstasks/retrieval_dataset_loaders.py) does NOT
enforce: it filters `queries` down to only the ids that appear in `qrels`
(load(), lines ~97-99), but never checks the reverse -- that every
qrel query-id actually has a matching row in `queries`, or that every qrel
corpus-id actually has a matching row in `corpus`. If a dataset's qrels
config references an id missing from queries/corpus, that silently produces
either a dropped qrel or a KeyError somewhere downstream in evaluation,
depending on the code path.

This is a different failure mode than the Clotho bug (PR #5062): Clotho's
empty-string queries still had *valid* ids and *valid* qrels pointing at
*existing* documents -- the defect was empty text content, not a dangling
reference. This script is a separate, from-scratch audit of a different
invariant, done before deciding whether it's worth turning into a permanent
check (mirrors the audit-first approach used for Clotho).

Deliberately avoids ever touching a media column (audio/image/video), so it
needs no torchcodec/ffmpeg: pulls only the "id" column from corpus/queries,
which HF `datasets` can do without decoding any Audio/Video feature in other
columns of the same file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

import mteb

OUTPUT_PATH = Path(__file__).parent / "qrel_id_membership_audit.json"

# Full de-duplicated retrieval-family task list across the four curated
# omni-modal benchmarks (MTEB(eng, v2), MIEB(lite), MAEB(beta), MVEB(beta)),
# derived from experiments/moeb_dataset_audit/moeb_task_inventory.json.
# "ClothoT2ARetrieval" replaced with "ClothoT2ARetrieval.v2" (the inventory
# snapshot predates the .v2 fix); "ClothoA2TRetrieval.v2" added even though
# not separately in the inventory, since it's the natural pair and already
# known-clean from the earlier pilot. Some of these (FEVER/HotpotQA/Climate
# FEVER-HardNegatives, standard BEIR tasks) have multi-million-row corpora --
# expect this run to take much longer than the earlier 12-task pilot, purely
# from download/IO time, not from any media decoding (we still only ever
# touch the "id" column).
TASK_NAMES = [
    "AVMemeExamAT2VRetrieval",
    "ActivityNetCaptionsT2VRetrieval",
    "ArguAna",
    "AskUbuntuDupQuestions",
    "AudioCapsAVVA2TRetrieval",
    "AudioCapsAVVT2ARetrieval",
    "AudioCapsT2ARetrieval",
    "AudioCapsA2TRetrieval",
    "CIRRIT2IRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "CUB200I2IRetrieval",
    "ClimateFEVERHardNegatives",
    "ClothoT2ARetrieval.v2",
    "ClothoA2TRetrieval.v2",
    "CommonVoiceMini21T2ARetrieval",
    "FEVERHardNegatives",
    "Fashion200kI2TRetrieval",
    "FiQA2018",
    "FleursT2ARetrieval",
    "GigaSpeechT2ARetrieval",
    "HatefulMemesI2TRetrieval",
    "HotpotQAHardNegatives",
    "InfoSeekIT2TRetrieval",
    "JamAltArtistA2ARetrieval",
    "JamAltLyricA2TRetrieval",
    "MACST2ARetrieval",
    "MSVDT2VRetrieval",
    "MindSmallReranking",
    "NIGHTSI2IRetrieval",
    "OVENIT2TRetrieval",
    "RP2kI2IRetrieval",
    "SCIDOCS",
    "SpokenSQuADT2ARetrieval",
    "TRECCOVID",
    "Touche2020Retrieval.v3",
    "UrbanSound8KT2ARetrieval",
    "VALOR32KT2VARetrieval",
    "VATEXV2ARetrieval",
    "VATEXVA2TRetrieval",
    "VGGSoundAVA2VRetrieval",
    "VQA2IT2TRetrieval",
    "VisualNewsI2TRetrieval",
    "WebQAT2ITRetrieval",
    "YouCook2T2VARetrieval",
]


def _resolve_split(hf_repo: str, revision: str, config: str, wanted_split: str) -> str:
    """Mirrors RetrievalDatasetLoader._get_split: exact match, else the lone split."""
    splits = get_dataset_split_names(hf_repo, revision=revision, config_name=config)
    if wanted_split in splits:
        return wanted_split
    if len(splits) == 1:
        return str(splits[0])
    raise ValueError(f"Split {wanted_split} not found in {splits} for config {config}")


def _ids_for_config(
    hf_repo: str, revision: str, config: str, wanted_split: str
) -> set[str]:
    split = _resolve_split(hf_repo, revision, config, wanted_split)
    ds = load_dataset(hf_repo, config, revision=revision, split=split)
    id_col = "_id" if "_id" in ds.column_names and "id" not in ds.column_names else "id"
    return {str(x) for x in ds[id_col]}


def _resolve_config(
    available: list[str], prefix: str | None, name: str, fallback: str | None
) -> str:
    candidate = f"{prefix}-{name}" if prefix else name
    if candidate in available:
        return candidate
    if fallback is not None:
        fallback_candidate = f"{prefix}-{fallback}" if prefix else fallback
        if fallback_candidate in available:
            return fallback_candidate
    raise ValueError(f"No config found for {name!r}/{fallback!r} among {available}")


def audit_task(task_name: str) -> dict[str, Any]:
    task = mteb.get_task(task_name)
    dataset_path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    split = task.metadata.eval_splits[0]
    hf_subsets = list(task.hf_subsets) if task.metadata.is_multilingual else ["default"]

    available = get_dataset_config_names(dataset_path, revision)

    per_subset: dict[str, Any] = {}
    for hf_subset in hf_subsets:
        prefix = None if hf_subset == "default" else hf_subset

        corpus_config = _resolve_config(available, prefix, "corpus", None)
        queries_config = _resolve_config(available, prefix, "queries", "query")
        qrels_config = _resolve_config(available, prefix, "default", "qrels")
        # "default" only resolves directly (no prefix trick needed) when unprefixed
        if qrels_config not in available and prefix is None:
            qrels_config = "qrels" if "qrels" in available else "default"

        corpus_ids = _ids_for_config(dataset_path, revision, corpus_config, split)
        query_ids = _ids_for_config(dataset_path, revision, queries_config, split)

        qrels_split = _resolve_split(dataset_path, revision, qrels_config, split)
        qrels_ds = load_dataset(
            dataset_path, qrels_config, revision=revision, split=qrels_split
        )
        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id"])
        qrel_query_ids = {str(x) for x in qrels_ds["query-id"]}
        qrel_corpus_ids = {str(x) for x in qrels_ds["corpus-id"]}

        missing_query_ids = qrel_query_ids - query_ids
        missing_corpus_ids = qrel_corpus_ids - corpus_ids

        per_subset[hf_subset] = {
            "num_corpus": len(corpus_ids),
            "num_queries": len(query_ids),
            "num_qrel_rows": len(qrels_ds),
            "num_missing_query_ids": len(missing_query_ids),
            "num_missing_corpus_ids": len(missing_corpus_ids),
            "example_missing_query_ids": sorted(missing_query_ids)[:5],
            "example_missing_corpus_ids": sorted(missing_corpus_ids)[:5],
        }

    return {"dataset": dataset_path, "revision": revision, "subsets": per_subset}


def main() -> None:
    results: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        print(f"\n=== {task_name} ===")
        try:
            result = audit_task(task_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc!r}")
            results[task_name] = {"error": repr(exc)}
            continue

        results[task_name] = result
        for hf_subset, stats in result["subsets"].items():
            flag = (
                "FOUND"
                if stats["num_missing_query_ids"] or stats["num_missing_corpus_ids"]
                else "clean"
            )
            print(
                f"  [{hf_subset}] corpus={stats['num_corpus']} "
                f"queries={stats['num_queries']} qrel_rows={stats['num_qrel_rows']} "
                f"missing_query_ids={stats['num_missing_query_ids']} "
                f"missing_corpus_ids={stats['num_missing_corpus_ids']} -> {flag}"
            )
            if stats["example_missing_query_ids"]:
                print(f"    example missing query ids: {stats['example_missing_query_ids']}")
            if stats["example_missing_corpus_ids"]:
                print(f"    example missing corpus ids: {stats['example_missing_corpus_ids']}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUTPUT_PATH}")

    total_found = sum(
        1
        for r in results.values()
        if "subsets" in r
        and any(
            s["num_missing_query_ids"] or s["num_missing_corpus_ids"]
            for s in r["subsets"].values()
        )
    )
    print(f"\nTasks with dangling qrel ids: {total_found} / {len(TASK_NAMES)}")


if __name__ == "__main__":
    main()
