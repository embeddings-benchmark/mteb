"""Create MIRACLRerankingDownsampled dataset from MIRACLRetrievalHardNegatives.

Creates a downsampled reranking task using the MIRACLRetrievalHardNegatives
corpus and generating top_ranked candidates per query via BM25.

Usage:
    python scripts/data/miracl_reranking/create_data.py
"""

from __future__ import annotations

import logging

import bm25s
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SOURCE_DATASET = "mteb/MIRACLRetrievalHardNegatives"
SOURCE_REVISION = "d7d94fa4b946cec4a27c84653aa0cf6b33f74a3c"
TARGET_REPO = "embedding-benchmark/MIRACLRerankingDownsampled"
SPLIT = "dev"
TOP_K = 250

LANGUAGES = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "sw",
    "te",
    "th",
    "yo",
    "zh",
]


def create_top_ranked_for_language(
    lang: str,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """Create reranking data for a single language.

    Loads the hard negatives corpus, queries, and qrels, then runs BM25 to
    find the top-K candidate documents per query. Returns corpus, queries,
    qrels, and top_ranked datasets.
    """
    logger.info("Processing language: %s", lang)

    corpus_ds = load_dataset(
        SOURCE_DATASET,
        f"{lang}-corpus",
        split=SPLIT,
        revision=SOURCE_REVISION,
    )
    queries_ds = load_dataset(
        SOURCE_DATASET,
        f"{lang}-queries",
        split=SPLIT,
        revision=SOURCE_REVISION,
    )
    qrels_ds = load_dataset(
        SOURCE_DATASET,
        f"{lang}-qrels",
        split=SPLIT,
        revision=SOURCE_REVISION,
    )

    # Build doc_id → index mapping
    corpus_ids = corpus_ds["id"]
    corpus_texts = corpus_ds["text"]
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    # Collect relevant doc IDs per query from qrels
    relevant_per_query: dict[str, set[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row["score"] > 0:
            relevant_per_query.setdefault(qid, set()).add(cid)

    # Tokenize corpus for BM25
    logger.info("[%s] Tokenizing %d corpus documents...", lang, len(corpus_ds))
    corpus_tokens = bm25s.tokenize(corpus_texts, show_progress=False)

    # Build BM25 index
    logger.info("[%s] Building BM25 index...", lang)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    # Run BM25 for each query
    query_ids = queries_ds["id"]
    query_texts = queries_ds["text"]

    logger.info("[%s] Running BM25 for %d queries...", lang, len(queries_ds))
    query_tokens = bm25s.tokenize(query_texts, show_progress=False)
    results, _ = retriever.retrieve(query_tokens, k=min(TOP_K, len(corpus_ds)))

    # Build top_ranked per query, ensuring relevant docs are always included
    top_ranked_rows = []
    used_doc_ids: set[str] = set()

    for q_idx, qid in enumerate(query_ids):
        qid = str(qid)
        bm25_doc_indices = results[q_idx].tolist()
        bm25_doc_ids = [corpus_ids[idx] for idx in bm25_doc_indices]

        # Start with BM25 results
        candidate_ids = list(dict.fromkeys(bm25_doc_ids))  # preserve order, deduplicate

        # Ensure all relevant docs are included
        relevant_ids = relevant_per_query.get(qid, set())
        for rel_id in relevant_ids:
            if rel_id not in candidate_ids and rel_id in id_to_idx:
                candidate_ids.append(rel_id)

        top_ranked_rows.append({"query-id": qid, "corpus-ids": candidate_ids})
        used_doc_ids.update(candidate_ids)

    # Filter corpus to only used documents
    filtered_corpus = corpus_ds.filter(
        lambda x: x["id"] in used_doc_ids, desc=f"[{lang}] Filtering corpus"
    )

    logger.info(
        "[%s] Corpus: %d → %d documents", lang, len(corpus_ds), len(filtered_corpus)
    )

    top_ranked_ds = Dataset.from_list(top_ranked_rows)

    return filtered_corpus, queries_ds, qrels_ds, top_ranked_ds


def main() -> None:
    ds_dict = DatasetDict()

    for lang in LANGUAGES:
        corpus, queries, qrels, top_ranked = create_top_ranked_for_language(lang)

        ds_dict[f"{lang}-corpus"] = corpus
        ds_dict[f"{lang}-queries"] = queries
        ds_dict[f"{lang}-qrels"] = qrels
        ds_dict[f"{lang}-top_ranked"] = top_ranked

    logger.info("Pushing dataset to %s...", TARGET_REPO)
    # Push each config separately since corpus/queries have different schemas
    for key, ds in ds_dict.items():
        logger.info("  Pushing config %s (%d rows)...", key, len(ds))
        ds.push_to_hub(TARGET_REPO, config_name=key, split="dev")
    logger.info("Done!")


if __name__ == "__main__":
    main()
