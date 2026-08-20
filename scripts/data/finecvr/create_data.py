from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download


SOURCE_REPO = "syn-omni-sony/finecvr"
SOURCE_REVISION = "410365ff0e457c5f8ed2d1814fe68a0ed2d1656f"


def load_annotations() -> list[dict]:
    path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="test.json",
        repo_type="dataset",
        revision=SOURCE_REVISION,
    )
    with open(path) as f:
        return json.load(f)


def build_metadata(data: list[dict]):
    corpus_ids = sorted(
        {row["source"] for row in data}
        | {row["target"] for row in data}
    )

    queries = []
    qrels = []
    top_ranked = []

    for i, row in enumerate(data):
        query_id = f"query-{i:05d}"
        source_id = row["source"]
        target_id = row["target"]

        queries.append(
            {
                "id": query_id,
                "text": row["instruct"],
                "source_id": source_id,
                "category": source_id.split("/", 1)[0],
            }
        )

        qrels.append(
            {
                "query-id": query_id,
                "corpus-id": target_id,
                "score": 1,
            }
        )

        # FineCVR evaluates against a global gallery while excluding
        # the query's own reference video from retrieval candidates.
        top_ranked.append(
            {
                "query-id": query_id,
                "corpus-ids": [
                    corpus_id
                    for corpus_id in corpus_ids
                    if corpus_id != source_id
                ],
            }
        )

    return corpus_ids, queries, qrels, top_ranked


def validate(
    corpus_ids: list[str],
    queries: list[dict],
    qrels: list[dict],
    top_ranked: list[dict],
) -> None:
    assert len(corpus_ids) == 2165
    assert len(queries) == 2000
    assert len(qrels) == 2000
    assert len(top_ranked) == 2000

    assert len(set(corpus_ids)) == len(corpus_ids)
    assert len({q["id"] for q in queries}) == len(queries)

    for query, qrel, ranked in zip(queries, qrels, top_ranked):
        assert query["id"] == qrel["query-id"]
        assert query["id"] == ranked["query-id"]
        assert query["source_id"] != qrel["corpus-id"]
        assert qrel["corpus-id"] in corpus_ids
        assert query["source_id"] in corpus_ids
        assert query["source_id"] not in ranked["corpus-ids"]
        assert qrel["corpus-id"] in ranked["corpus-ids"]
        assert len(ranked["corpus-ids"]) == 2164




def validate_media(corpus_ids: list[str], tar_path: str) -> None:
    needed = set(corpus_ids)
    counts: Counter[str] = Counter()

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue

            parts = member.name.split("/")
            if len(parts) < 4:
                continue

            clip_id = "/".join(parts[1:3])
            if clip_id in needed:
                counts[clip_id] += 1

    missing = sorted(needed - counts.keys())
    wrong_frame_count = sorted(
        (clip_id, counts[clip_id])
        for clip_id in needed
        if counts[clip_id] != 8
    )

    assert not missing, f"Missing benchmark clips: {missing[:10]}"
    assert not wrong_frame_count, (
        f"Expected exactly 8 frames per benchmark clip: "
        f"{wrong_frame_count[:10]}"
    )


def build_datasets(
    corpus_ids: list[str],
    queries: list[dict],
    qrels: list[dict],
    top_ranked: list[dict],
) -> dict[str, Dataset]:
    corpus_ds = Dataset.from_list(
        [{"id": corpus_id} for corpus_id in corpus_ids]
    )
    queries_ds = Dataset.from_list(queries)
    qrels_ds = Dataset.from_list(qrels)
    top_ranked_ds = Dataset.from_list(top_ranked)

    return {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels": qrels_ds,
        "top_ranked": top_ranked_ds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-path",
        required=True,
        help="Path to the downloaded FineCVR finecvr.tar.gz archive.",
    )
    args = parser.parse_args()

    data = load_annotations()
    corpus_ids, queries, qrels, top_ranked = build_metadata(data)

    validate(corpus_ids, queries, qrels, top_ranked)
    validate_media(corpus_ids, args.tar_path)
    datasets = build_datasets(corpus_ids, queries, qrels, top_ranked)

    print(f"corpus: {len(corpus_ids)}")
    print(f"queries: {len(queries)}")
    print(f"qrels: {len(qrels)}")
    print(f"top_ranked: {len(top_ranked)}")
    print(f"candidates/query: {len(top_ranked[0]['corpus-ids'])}")
    print()
    print("query example:", queries[0])
    print("qrel example:", qrels[0])
    print("top-ranked first 5:", top_ranked[0]["corpus-ids"][:5])
    print()
    for name, dataset in datasets.items():
        print(f"{name} features:", dataset.features)

    print()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
