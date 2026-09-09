#!/usr/bin/env python3
"""Build SpokenCOCO A2IT: audio → (image + text) retrieval for MTEB.

Takes the existing SpokenCOCO A2I task (audio → image) and adds MSCOCO text
captions to each corpus item, making the corpus trimodal (image + text).
Queries and qrels are identical to SpokenCOCO A2I; only the corpus gains a
'text' column. This creates a genuine audio+image+text retrieval task that
tests whether omni models can outperform unimodal specialists.

Sources:
  - Queries + qrels: whybe-choi/SpokenCOCOA2IRetrieval (HF)
  - Corpus images:   whybe-choi/SpokenCOCOA2IRetrieval (HF)
  - Corpus text:     jxie/coco_captions (HF, first caption per image)

Examples:
  # Dry run — print stats, push nothing.
  uv run python scripts/data/spokencoco_a2it/create_data.py --dry-run

  # Push to HuggingFace.
  HF_TOKEN=... uv run python scripts/data/spokencoco_a2it/create_data.py --push
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Image, Value
from huggingface_hub import HfApi, create_repo, get_token, hf_hub_download, list_repo_files

_SOURCE_A2I = "whybe-choi/SpokenCOCOA2IRetrieval"
_SOURCE_CAPTIONS = "jxie/coco_captions"
_TARGET_REPO = "rakshi719/SpokenCOCO-A2IT"
_LICENSE = "cc-by-4.0"


def _load_a2i_corpus() -> pd.DataFrame:
    """Load corpus parquets from SpokenCOCO A2I (id + image bytes)."""
    files = [
        f for f in list_repo_files(_SOURCE_A2I, repo_type="dataset")
        if f.startswith("corpus/") and f.endswith(".parquet")
    ]
    parts = []
    for f in sorted(files):
        path = hf_hub_download(_SOURCE_A2I, f, repo_type="dataset")
        parts.append(pd.read_parquet(path))
    df = pd.concat(parts, ignore_index=True)
    # id is zero-padded 12-char string like '000000391895'
    df["cocoid"] = df["id"].astype(int)
    return df


def _load_a2i_queries() -> pd.DataFrame:
    """Load query parquets from SpokenCOCO A2I (id + audio bytes)."""
    files = [
        f for f in list_repo_files(_SOURCE_A2I, repo_type="dataset")
        if f.startswith("queries/") and f.endswith(".parquet")
    ]
    parts = []
    for f in sorted(files):
        path = hf_hub_download(_SOURCE_A2I, f, repo_type="dataset")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def _load_a2i_qrels() -> pd.DataFrame:
    """Load qrels from SpokenCOCO A2I."""
    files = [
        f for f in list_repo_files(_SOURCE_A2I, repo_type="dataset")
        if f.startswith("qrels/") and f.endswith(".parquet")
    ]
    parts = []
    for f in sorted(files):
        path = hf_hub_download(_SOURCE_A2I, f, repo_type="dataset")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def _load_coco_captions() -> pd.DataFrame:
    """Load MSCOCO captions (test split), keep first caption per image."""
    files = [
        f for f in list_repo_files(_SOURCE_CAPTIONS, repo_type="dataset")
        if f.startswith("data/test-") and f.endswith(".parquet")
    ]
    parts = []
    for f in sorted(files):
        path = hf_hub_download(_SOURCE_CAPTIONS, f, repo_type="dataset")
        df = pd.read_parquet(path, columns=["cocoid", "caption"])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # Keep first caption per image (MSCOCO has 5 per image)
    return df.drop_duplicates(subset="cocoid", keep="first").reset_index(drop=True)


def _build_corpus(corpus_df: pd.DataFrame, captions_df: pd.DataFrame) -> tuple[Dataset, int]:
    """Join images with captions, return trimodal corpus dataset."""
    merged = corpus_df.merge(captions_df, on="cocoid", how="inner")
    n_matched = len(merged)
    n_unmatched = len(corpus_df) - n_matched
    if n_unmatched > 0:
        print(f"  Warning: {n_unmatched} corpus images had no MSCOCO caption")

    corpus = Dataset.from_dict({
        "id": merged["id"].tolist(),
        "image": merged["image"].tolist(),
        "text": merged["caption"].tolist(),
    }).cast_column("image", Image())
    return corpus, n_matched


def _build_queries(queries_df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "id": queries_df["id"].tolist(),
        "audio": queries_df["audio"].tolist(),
    }).cast_column("audio", Audio())


def _build_qrels(qrels_df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "query-id": qrels_df["query-id"].tolist(),
        "corpus-id": qrels_df["corpus-id"].tolist(),
        "score": qrels_df["score"].tolist(),
    }).cast_column("score", Value("int32"))


def _dataset_card(n_corpus: int, n_queries: int, n_qrels: int) -> str:
    return f"""---
license: {_LICENSE}
pretty_name: SpokenCOCO Audio-to-Image+Text Retrieval
tags:
- mteb
- moeb
- audio-to-image-text
- cross-modal-retrieval
- spoken-language
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
---

# SpokenCOCO Audio-to-(Image+Text) Retrieval

MTEB/MOEB task where queries are spoken audio captions and corpus items
contain both a MSCOCO image **and** its written text caption.

## Task

Given a spoken audio description of an image, retrieve the correct
(image, text) pair from the corpus. Only models that can process all three
modalities — audio, image, and text — can exploit the full corpus signal.

## Contents

- **Queries**: {n_queries} spoken audio captions (WAV, ~16kHz)
- **Corpus**: {n_corpus} (image, text) pairs from MS-COCO
- **Qrels**: {n_qrels} binary relevance judgments

## Construction

Audio queries and qrels are taken directly from the SpokenCOCO A2I task
(`{_SOURCE_A2I}`). The corpus images come from the same source; MSCOCO text
captions are joined from `{_SOURCE_CAPTIONS}` (first caption per image).

## License

CC-BY-4.0 (inherits from MS-COCO and SpokenCOCO).

## Citation

```bibtex
@inproceedings{{shih2023spokencoco,
  title={{Speechclip: Integrating speech encoder and large vision-language
         model for spoken language understanding}},
  author={{Shih, Yi-Jen and others}},
  booktitle={{ASRU}},
  year={{2023}},
}}
@misc{{lin2014coco,
  title={{Microsoft COCO: Common Objects in Context}},
  author={{Lin, Tsung-Yi and others}},
  year={{2014}},
  eprint={{1405.0312}},
}}
```
"""


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-id", default=_TARGET_REPO)
    args = parser.parse_args()

    print("Loading SpokenCOCO A2I corpus...")
    corpus_df = _load_a2i_corpus()
    print(f"  {len(corpus_df)} corpus images")

    print("Loading MSCOCO captions...")
    captions_df = _load_coco_captions()
    print(f"  {len(captions_df)} unique images with captions")

    print("Loading SpokenCOCO A2I queries...")
    queries_df = _load_a2i_queries()
    print(f"  {len(queries_df)} audio queries")

    print("Loading SpokenCOCO A2I qrels...")
    qrels_df = _load_a2i_qrels()
    print(f"  {len(qrels_df)} qrels")

    print("Building trimodal corpus (image + text)...")
    corpus, n_matched = _build_corpus(corpus_df, captions_df)
    print(f"  {n_matched} corpus items with image + text")

    queries = _build_queries(queries_df)
    qrels = _build_qrels(qrels_df)

    # Filter qrels to only include corpus items that got captions
    matched_ids = set(corpus["id"])
    qrels_df_filtered = qrels_df[qrels_df["corpus-id"].isin(matched_ids)]
    qrels = _build_qrels(qrels_df_filtered)

    print(f"\nFinal stats:")
    print(f"  corpus:  {len(corpus)} items (image + text)")
    print(f"  queries: {len(queries)} audio queries")
    print(f"  qrels:   {len(qrels)}")

    if args.dry_run:
        print("\nDry run — done.")
        return

    if args.push:
        token = get_token() or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("No HuggingFace token found")

        print(f"\nPushing to {args.repo_id}...")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=_dataset_card(len(corpus), len(queries), len(qrels)).encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        DatasetDict({"test": corpus}).push_to_hub(
            args.repo_id, "corpus", token=token, max_shard_size="500MB",
            commit_message="Add trimodal corpus (image + text)",
        )
        DatasetDict({"test": queries}).push_to_hub(
            args.repo_id, "queries", token=token, max_shard_size="500MB",
            commit_message="Add audio queries",
        )
        DatasetDict({"test": qrels}).push_to_hub(
            args.repo_id, "qrels", token=token,
            commit_message="Add qrels",
        )
        revision = api.dataset_info(args.repo_id).sha
        print(f"Pushed {args.repo_id} @ {revision}")
        Path("hub_revision_a2it.txt").write_text(f"{revision}\n")


if __name__ == "__main__":
    main()
