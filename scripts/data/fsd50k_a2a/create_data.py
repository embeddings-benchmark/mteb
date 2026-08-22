"""Build the FSD50KA2ARetrieval audio-to-audio retrieval task from FSD50K.

Source: mteb/fsd50k_mini (test split), the same permissively-licensed FSD50K
subset already used for the FSD50K multilabel classification task in mteb
(mteb/tasks/multilabel_classification/eng/fsd50_hf.py). That split mirrors
FSD50K's own public eval split.

Each clip's `labels` field is a comma-separated list of AudioSet-ontology
tags ordered from most to least specific (e.g.
"Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music").
The first tag is treated as the clip's primary class -- this avoids the
overly broad relevance judgments that using the full multi-label hierarchy
would produce (two clips sharing only a distant ancestor tag like "Music"
should not be judged relevant to each other).

From the primary-class groups with at least QUERIES_PER_CLASS +
CORPUS_PER_CLASS clips, NUM_CLASSES classes are selected and, per class,
QUERIES_PER_CLASS clips become queries and a disjoint CORPUS_PER_CLASS
clips become corpus documents. Relevance (qrels) is same-class membership:
every query is relevant to every corpus document of its own class.

    100 queries + 200 corpus docs (20 classes x 5 queries x 10 corpus)
    1000 qrels rows (20 classes x 5 queries x 10 corpus docs each)

Usage:
    python scripts/data/fsd50k_a2a/create_data.py --push-to-hub yaswanth169/FSD50K-A2ARetrieval
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset

SOURCE_DATASET = "mteb/fsd50k_mini"
SOURCE_SPLIT = "test"
NUM_CLASSES = 20
QUERIES_PER_CLASS = 5
CORPUS_PER_CLASS = 10
SEED = 42


def primary_label(labels: str) -> str:
    return labels.split(",")[0]


def build_splits(seed: int = SEED) -> tuple[Dataset, Dataset, list[dict]]:
    source = load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)

    by_class: dict[str, list[int]] = defaultdict(list)
    for idx, labels in enumerate(source["labels"]):
        by_class[primary_label(labels)].append(idx)

    min_per_class = QUERIES_PER_CLASS + CORPUS_PER_CLASS
    eligible = sorted(
        (cls for cls, idxs in by_class.items() if len(idxs) >= min_per_class),
        key=lambda cls: (-len(by_class[cls]), cls),
    )[:NUM_CLASSES]
    assert len(eligible) == NUM_CLASSES, (
        f"only {len(eligible)} classes have >= {min_per_class} clips"
    )

    rng = random.Random(seed)
    query_rows: list[dict] = []
    corpus_rows: list[dict] = []
    qrels: list[dict] = []
    for cls in eligible:
        idxs = list(by_class[cls])
        rng.shuffle(idxs)
        query_idxs = idxs[:QUERIES_PER_CLASS]
        corpus_idxs = idxs[QUERIES_PER_CLASS : QUERIES_PER_CLASS + CORPUS_PER_CLASS]

        query_ids = [f"q-{cls}-{i}" for i in range(len(query_idxs))]
        corpus_ids = [f"c-{cls}-{i}" for i in range(len(corpus_idxs))]

        for qid, idx in zip(query_ids, query_idxs):
            query_rows.append({"id": qid, "audio": source[idx]["audio"]})
        for cid, idx in zip(corpus_ids, corpus_idxs):
            corpus_rows.append({"id": cid, "audio": source[idx]["audio"]})
        for qid in query_ids:
            for cid in corpus_ids:
                qrels.append({"query-id": qid, "corpus-id": cid, "score": 1})

    queries = Dataset.from_list(query_rows)
    corpus = Dataset.from_list(corpus_rows)
    return queries, corpus, qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", default=None, help="HF repo id to push to")
    args = parser.parse_args()

    queries, corpus, qrels = build_splits()
    qrels_ds = Dataset.from_list(qrels)

    print(f"queries: {len(queries)}, corpus: {len(corpus)}, qrels: {len(qrels_ds)}")
    assert len(queries) == NUM_CLASSES * QUERIES_PER_CLASS
    assert len(corpus) == NUM_CLASSES * CORPUS_PER_CLASS
    assert len(qrels_ds) == NUM_CLASSES * QUERIES_PER_CLASS * CORPUS_PER_CLASS
    assert set(queries["id"]).isdisjoint(set(corpus["id"]))

    if args.push_to_hub:
        DatasetDict({"test": queries}).push_to_hub(args.push_to_hub, "queries")
        DatasetDict({"test": corpus}).push_to_hub(args.push_to_hub, "corpus")
        DatasetDict({"test": qrels_ds}).push_to_hub(args.push_to_hub, "qrels")


if __name__ == "__main__":
    main()
