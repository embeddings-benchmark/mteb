"""Normalize dianetc/OBLIQ-Bench into MTEB retrieval format on mteb/OBLIQBenchRetrieval."""

from __future__ import annotations

import csv
import json
import logging

from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_REPO = "dianetc/OBLIQ-Bench"
SOURCE_REVISION = "4ebee29f68ceeb62ca00bd73d5478acdd3cd7764"
TARGET_REPO = "mteb/OBLIQBenchRetrieval"

SUBSET_PREFIXES: dict[str, str] = {
    "twitter": "descriptive/twitter",
    "wildchat": "descriptive/wildchat",
    "math": "analogues/math",
    "writing": "analogues/writing",
    "congress": "tip-of-tongue/congress",
}

# Subsets that ship per_query_excluded_ids.json (same-source docs to drop per query).
EXCLUSION_SUBSETS = ("math", "writing")

# Subsets that ship qrels_pool.tsv (pooled + judged superset of gold qrels.tsv).
# Writing and congress only ship gold qrels, so fall back to that.
POOLED_QRELS_SUBSETS = ("twitter", "wildchat", "math")

CORPUS_FEATURES = Features({"id": Value("string"), "text": Value("string")})
QUERIES_FEATURES = Features({"id": Value("string"), "text": Value("string")})
QRELS_FEATURES = Features(
    {
        "query-id": Value("string"),
        "corpus-id": Value("string"),
        "score": Value("int32"),
    }
)
TOP_RANKED_FEATURES = Features(
    {
        "query-id": Value("string"),
        "corpus-ids": Sequence(Value("string")),
    }
)


def _download(prefix: str, rel_path: str) -> str:
    return hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=f"{prefix}/{rel_path}",
        revision=SOURCE_REVISION,
        repo_type="dataset",
    )


def _load_jsonl_with_id(path: str) -> Dataset:
    ds = Dataset.from_json(path)
    if "_id" in ds.column_names and "id" not in ds.column_names:
        ds = ds.rename_column("_id", "id")
    ds = ds.cast_column("id", Value("string"))
    return ds.select_columns(["id", "text"])


def _read_qrels(path: str) -> Dataset:
    rows: list[dict[str, str | int]] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        normalized = [h.replace("_", "-") for h in header]
        if normalized != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels header in {path}: {header}")
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"Malformed qrels row in {path}: {row}")
            rows.append({"query-id": row[0], "corpus-id": row[1], "score": int(row[2])})
    return Dataset.from_list(rows, features=QRELS_FEATURES)


def _build_top_ranked(path: str, corpus_ids: list[str]) -> Dataset:
    """Convert per_query_excluded_ids.json into a top_ranked dataset.

    Output rows are (query-id, corpus-ids) where corpus-ids is the candidate
    pool for that query: the full corpus minus same-source exclusions.
    """
    with open(path, encoding="utf-8") as fh:
        excluded_by_qid: dict[str, list[str]] = json.load(fh)
    rows = []
    for qid, excluded in excluded_by_qid.items():
        excl_set = set(excluded)
        rows.append(
            {
                "query-id": qid,
                "corpus-ids": [cid for cid in corpus_ids if cid not in excl_set],
            }
        )
    return Dataset.from_list(rows, features=TOP_RANKED_FEATURES)


def _push(ds: Dataset, config_name: str, split: str = "test") -> None:
    logger.info(
        "Pushing config %s (split=%s, rows=%d) to %s",
        config_name,
        split,
        len(ds),
        TARGET_REPO,
    )
    ds.push_to_hub(TARGET_REPO, config_name=config_name, split=split)


def process_subset(subset: str, prefix: str) -> None:
    logger.info("Processing subset %s (%s)", subset, prefix)

    corpus_path = _download(prefix, "corpus/corpus.jsonl")
    corpus = _load_jsonl_with_id(corpus_path).cast(CORPUS_FEATURES)
    _push(corpus, f"{subset}-corpus")

    queries_path = _download(prefix, "queries+qrels/queries.jsonl")
    queries = _load_jsonl_with_id(queries_path).cast(QUERIES_FEATURES)
    _push(queries, f"{subset}-queries")

    # qrels_pool.tsv is the pooled judgment set: original gold qrels plus
    # extra docs surfaced by retrievers and judged during pooling. Dataset
    # author recommends evaluating on the pool over the sparser gold qrels.
    # Only twitter/wildchat/math ship a pool; writing/congress use gold qrels.
    qrels_filename = "qrels_pool.tsv" if subset in POOLED_QRELS_SUBSETS else "qrels.tsv"
    qrels_path = _download(prefix, f"queries+qrels/{qrels_filename}")
    qrels = _read_qrels(qrels_path)
    logger.info("  Using %s for %s (%d judgments)", qrels_filename, subset, len(qrels))
    _push(qrels, f"{subset}-qrels")

    if subset in EXCLUSION_SUBSETS:
        excluded_path = _download(prefix, "queries+qrels/per_query_excluded_ids.json")
        corpus_ids = list(corpus["id"])
        top_ranked = _build_top_ranked(excluded_path, corpus_ids)
        _push(top_ranked, f"{subset}-top_ranked")


def main() -> None:
    for subset, prefix in SUBSET_PREFIXES.items():
        process_subset(subset, prefix)
    pushed = sum(4 if s in EXCLUSION_SUBSETS else 3 for s in SUBSET_PREFIXES)
    logger.info("Done. Pushed %d configs.", pushed)


if __name__ == "__main__":
    main()
