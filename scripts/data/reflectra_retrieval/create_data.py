#!/usr/bin/env python3
"""Package AraNge/reflectra-benchmark into MTEB image→audio retrieval.

Reflectra contains 1,000 images each rated against six candidate music clips on a
0–10 scale. We treat scores >= 7 as relevant for binary qrels (image query →
audio corpus).

The upstream audio table reuses identical clips under different audio_ids; the
corpus is therefore deduplicated by content hash and qrels are remapped to the
canonical ids.

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
import hashlib
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


def _media_bytes(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw:
            return bytes(raw)
        path = cell.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def _media_hash(cell) -> str | None:
    raw = _media_bytes(cell)
    if not raw:
        return None
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


def _dedupe_audios(
    audio_ids: list,
    audios: list,
) -> tuple[list, list, dict]:
    """Keep one corpus row per unique audio content; map aliases → canonical id."""
    hash_to_id: dict[str, object] = {}
    id_remap: dict[object, object] = {}
    canonical_ids: list = []
    canonical_audios: list = []

    for audio_id, audio in zip(audio_ids, audios, strict=True):
        media_hash = _media_hash(audio)
        if media_hash is None:
            continue
        canonical_id = hash_to_id.get(media_hash)
        if canonical_id is None:
            hash_to_id[media_hash] = audio_id
            canonical_id = audio_id
            canonical_ids.append(audio_id)
            canonical_audios.append(audio)
        id_remap[audio_id] = canonical_id

    return canonical_ids, canonical_audios, id_remap


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

    corpus_ids, corpus_audios, audio_id_remap = _dedupe_audios(
        audios["audio_id"],
        audios["audio"],
    )
    dropped = len(audios["audio_id"]) - len(corpus_ids)
    if dropped:
        print(
            f"Deduped corpus audios: {len(audios['audio_id'])} → {len(corpus_ids)} "
            f"({dropped} duplicate contents removed)"
        )

    i2a_corpus = Dataset.from_dict(
        {"id": corpus_ids, "audio": corpus_audios},
    ).cast_column("audio", Audio())

    canonical_ids = set(corpus_ids)
    qrels = {"query-id": [], "corpus-id": [], "score": []}
    seen_pairs: set[tuple[object, object]] = set()
    skipped_queries = 0
    for qid, audio_ids, score_list in zip(
        scores["image_id"],
        scores["audio_ids"],
        scores["scores"],
        strict=True,
    ):
        added = False
        for aid, score in zip(audio_ids, score_list, strict=True):
            if score < args.score_threshold:
                continue
            canonical_aid = audio_id_remap.get(aid)
            if canonical_aid is None or canonical_aid not in canonical_ids:
                continue
            pair = (qid, canonical_aid)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            qrels["query-id"].append(qid)
            qrels["corpus-id"].append(canonical_aid)
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
