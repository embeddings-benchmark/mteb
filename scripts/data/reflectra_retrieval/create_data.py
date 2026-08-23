#!/usr/bin/env python3
"""Package AraNge/reflectra-benchmark into MTEB image→audio retrieval.

Reflectra contains 1,000 images each rated against six candidate music clips on a
0–10 scale. We treat scores >= 7 as relevant for binary qrels (image query →
audio corpus).

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/reflectra_retrieval/create_data.py \\
      --repo-id Wissam42/Reflectra-I2A \\
      --work-dir /tmp/reflectra_mteb \\
      --score-threshold 7 \\
      --push
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Dataset, DatasetDict, Image
from huggingface_hub import create_repo, hf_hub_download

_SOURCE = "AraNge/reflectra-benchmark"


def _push_retrieval_direction(
    *,
    repo_id: str,
    queries: Dataset,
    corpus: Dataset,
    qrels: Dataset,
    token: str | None,
    out_dir: Path | None,
) -> None:
    if token:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
        DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
        DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)
        print(f"Pushed {repo_id}")
        return

    assert out_dir is not None
    out = out_dir / repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    corpus.save_to_disk(out / "corpus")
    queries.save_to_disk(out / "queries")
    qrels.save_to_disk(out / "qrels")
    print(f"Wrote {out}")


def _download(filename: str) -> str:
    # Without repo_type="dataset", hf_hub_download looks under models/ and 404s.
    return hf_hub_download(_SOURCE, filename, repo_type="dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Wissam42/Reflectra-I2A")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/reflectra_mteb"))
    parser.add_argument("--score-threshold", type=int, default=7)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    scores_path = _download("image_audio_scores.parquet")
    image_path = _download("image_table.parquet")
    audio_path = _download("audio_table.parquet")

    scores = pq.read_table(scores_path).to_pydict()
    images = pq.read_table(image_path).to_pydict()
    audios = pq.read_table(audio_path).to_pydict()

    i2a_queries = Dataset.from_dict(
        {"id": images["image_id"], "image": images["image"]},
    ).cast_column("image", Image())

    corpus_ids = audios["audio_id"]
    i2a_corpus = Dataset.from_dict(
        {"id": corpus_ids, "audio": audios["audio"]},
    ).cast_column("audio", Audio())

    audio_by_id = set(audios["audio_id"])
    qrels = {"query-id": [], "corpus-id": [], "score": []}
    skipped_queries = 0
    for qid, audio_ids, score_list in zip(
        scores["image_id"],
        scores["audio_ids"],
        scores["scores"],
        strict=True,
    ):
        added = False
        for aid, score in zip(audio_ids, score_list, strict=True):
            if score >= args.score_threshold and aid in audio_by_id:
                qrels["query-id"].append(qid)
                qrels["corpus-id"].append(aid)
                qrels["score"].append(1)
                added = True
        if not added:
            skipped_queries += 1

    if skipped_queries:
        print(f"Warning: {skipped_queries} queries have no positives at threshold")
    print(
        f"I2A corpus={len(i2a_corpus)} queries={len(i2a_queries)} "
        f"qrels={len(qrels['query-id'])} threshold>={args.score_threshold}"
    )

    token = os.environ.get("HF_TOKEN") if args.push else None
    if args.push and not token:
        raise SystemExit("Set HF_TOKEN to push")

    out = None if args.push else work / "mteb_export"
    _push_retrieval_direction(
        repo_id=args.repo_id,
        queries=i2a_queries,
        corpus=i2a_corpus,
        qrels=Dataset.from_dict(qrels),
        token=token,
        out_dir=out,
    )


if __name__ == "__main__":
    main()
