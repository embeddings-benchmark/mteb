"""
Final downsampling script for MIRACLReranking.

Strategy: Simple truncation of top_ranked from 100 to 60 candidates per query.
Qrels and corpus are filtered to match the retained candidates.

This achieves <2% NDCG@10 delta across tested models (BM25, e5-small) and languages (ar, en, fr, zh).
Corpus size is reduced by ~40%.
"""

import os
import json
from datasets import load_dataset, Dataset

REVISION = "d11a14c74e8bd448cedab0c1d9a720040535f228"
DATASET_NAME = "mteb/MIRACLReranking"
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
TOP_N = 70
OUTPUT_DIR = "/mnt/data/zoltan/miracl_comparison/MIRACLReranking_downsampled_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

stats = {}

for lang in LANGUAGES:
    print(f"Processing {lang}...")

    corpus = load_dataset(
        DATASET_NAME, f"{lang}-corpus", revision=REVISION, split="dev"
    )
    queries = load_dataset(
        DATASET_NAME, f"{lang}-queries", revision=REVISION, split="dev"
    )
    qrels = load_dataset(DATASET_NAME, f"{lang}-qrels", revision=REVISION, split="dev")
    top_ranked = load_dataset(
        DATASET_NAME, f"{lang}-top_ranked", revision=REVISION, split="dev"
    )

    # Truncate top_ranked to TOP_N candidates
    truncated_rows = []
    retained_cids = set()
    for row in top_ranked:
        trunc = row["corpus-ids"][:TOP_N]
        truncated_rows.append({"query-id": row["query-id"], "corpus-ids": trunc})
        retained_cids.update(trunc)

    new_top_ranked = Dataset.from_list(truncated_rows)

    # Filter qrels to retained candidates
    new_qrels = Dataset.from_list(
        [
            {
                "query-id": r["query-id"],
                "corpus-id": r["corpus-id"],
                "score": r["score"],
            }
            for r in qrels
            if r["corpus-id"] in retained_cids
        ]
    )

    # Filter corpus to retained candidates
    corpus_dict = {r["_id"]: r for r in corpus}
    new_corpus = Dataset.from_list(
        [corpus_dict[cid] for cid in sorted(retained_cids) if cid in corpus_dict]
    )

    # Save
    lang_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    new_corpus.save_to_disk(os.path.join(lang_dir, "corpus"))
    queries.save_to_disk(os.path.join(lang_dir, "queries"))
    new_qrels.save_to_disk(os.path.join(lang_dir, "qrels"))
    new_top_ranked.save_to_disk(os.path.join(lang_dir, "top_ranked"))

    pos_full = sum(1 for r in qrels if r["score"] > 0)
    pos_kept = sum(1 for r in new_qrels if r["score"] > 0)

    stats[lang] = {
        "queries": len(queries),
        "corpus_full": len(corpus),
        "corpus_down": len(new_corpus),
        "positives_full": pos_full,
        "positives_kept": pos_kept,
        "positives_pct": round(100 * pos_kept / pos_full, 1) if pos_full > 0 else 0,
    }

    print(
        f"  corpus: {len(corpus)} -> {len(new_corpus)} ({100 * len(new_corpus) / len(corpus):.0f}%), positives: {pos_full} -> {pos_kept} ({stats[lang]['positives_pct']}%)"
    )

with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}")
print(
    f"\nTotal corpus reduction: {sum(s['corpus_full'] for s in stats.values())} -> {sum(s['corpus_down'] for s in stats.values())}"
)
