#!/usr/bin/env python3
"""Sample SEACrowd/sea-vl_crawling into MTEB t2i / i2t retrieval Hub datasets.

Source is ~1.27M image–caption pairs (~109GB). This script stream-shuffles with
a fixed seed and keeps 2048 unique pairs (first non-empty caption each; duplicate
source ``id`` values are skipped), then pushes separate T2I and I2T repos in the
standard corpus / queries / qrels layout.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/sea_vl_crawling/create_data.py \\
      --t2i-repo-id {base_repo}/SEA-VL-Crawling-T2I \\
      --i2t-repo-id {base_repo}/SEA-VL-Crawling-I2T \\
      --push
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, DatasetDict, Image, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm

_SOURCE = "SEACrowd/sea-vl_crawling"
_SOURCE_REVISION = "4723b4f00b68dc2fe649624628f3e8d50afa1e74"
_N_SAMPLES = 2048
_SHUFFLE_BUFFER = 10_000
_SEED = 42


def _first_caption(captions: list[str] | None) -> str | None:
    if not captions:
        return None
    for cap in captions:
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return None


def _sample_pairs(n_samples: int) -> list[dict]:
    stream = load_dataset(
        _SOURCE,
        revision=_SOURCE_REVISION,
        split="train",
        streaming=True,
    )
    stream = stream.shuffle(seed=_SEED, buffer_size=_SHUFFLE_BUFFER)

    pairs: list[dict] = []
    seen_ids: set[str] = set()
    for row in tqdm(stream, total=n_samples, desc="sample"):
        pair_id = str(row["id"])
        if pair_id in seen_ids:
            continue
        text = _first_caption(row.get("caption"))
        image = row.get("image")
        if text is None or image is None:
            continue
        seen_ids.add(pair_id)
        pairs.append(
            {
                "id": pair_id,
                "image": image,
                "text": text,
                "category": row.get("category") or "",
            }
        )
        if len(pairs) >= n_samples:
            break
    if len(pairs) < n_samples:
        raise SystemExit(f"Only collected {len(pairs)} pairs; expected {n_samples}")
    return pairs


def _push_t2i(pairs: list[dict], repo_id: str, token: str) -> None:
    corpus = Dataset.from_list(
        [
            {"id": f"d-{p['id']}", "image": p["image"], "modality": "image"}
            for p in pairs
        ]
    ).cast_column("image", Image())
    queries = Dataset.from_list(
        [{"id": f"q-{p['id']}", "text": p["text"], "modality": "text"} for p in pairs]
    )
    qrels = Dataset.from_list(
        [
            {"query-id": f"q-{p['id']}", "corpus-id": f"d-{p['id']}", "score": 1}
            for p in pairs
        ]
    )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
    DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
    DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)


def _push_i2t(pairs: list[dict], repo_id: str, token: str) -> None:
    corpus = Dataset.from_list(
        [{"id": f"d-{p['id']}", "text": p["text"], "modality": "text"} for p in pairs]
    )
    queries = Dataset.from_list(
        [
            {"id": f"q-{p['id']}", "image": p["image"], "modality": "image"}
            for p in pairs
        ]
    ).cast_column("image", Image())
    qrels = Dataset.from_list(
        [
            {"query-id": f"q-{p['id']}", "corpus-id": f"d-{p['id']}", "score": 1}
            for p in pairs
        ]
    )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
    DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
    DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2i-repo-id", default="Wissam42/SEA-VL-Crawling-T2I")
    parser.add_argument("--i2t-repo-id", default="Wissam42/SEA-VL-Crawling-I2T")
    parser.add_argument("--n-samples", type=int, default=_N_SAMPLES)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--directions",
        default="both",
        choices=("t2i", "i2t", "both"),
        help="Which Hub datasets to build",
    )
    args = parser.parse_args()

    pairs = _sample_pairs(args.n_samples)
    print(
        f"sampled={len(pairs)} source={_SOURCE}@{_SOURCE_REVISION} "
        f"seed={_SEED} buffer={_SHUFFLE_BUFFER}"
    )

    if not args.push:
        print("Dry run (pass --push to upload). Example caption:")
        print(f"  id={pairs[0]['id']} cat={pairs[0]['category']!r}")
        print(f"  text={pairs[0]['text'][:200]!r}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to push")

    if args.directions in ("t2i", "both"):
        print(f"Pushing T2I → {args.t2i_repo_id}")
        _push_t2i(pairs, args.t2i_repo_id, token)
    if args.directions in ("i2t", "both"):
        print(f"Pushing I2T → {args.i2t_repo_id}")
        _push_i2t(pairs, args.i2t_repo_id, token)
    print("done — update task metadata revisions if Hub SHAs changed")


if __name__ == "__main__":
    main()
