"""Build ABOV2IRetrieval, the video-to-image direction of ABOI2VRetrieval.

Source: ``hubxrt/ABO-I2V`` (CC BY 4.0), built from Amazon Berkeley Objects by
``scripts/data/abo_i2v_retrieval/create_data.py``. Read that script's docstring
for how the spins and catalog photographs were selected.

Construction
------------
No media is re-selected or re-encoded. The ABO-I2V configs are swapped:

  queries = the 2,857 turntable spin videos (ABO-I2V ``corpus``)
  corpus  = the 2,857 in-situ catalog photographs (ABO-I2V ``queries``)
  qrels   = ABO-I2V qrels with ``query-id`` and ``corpus-id`` swapped, score 1

The task is "find the catalog listing photo for this product video". Each
product has exactly one eligible catalog photograph in ABO-I2V, so qrels stay
1:1. The photograph is an independent capture (different time, place and
lighting), never a frame of the spin.

Usage:
  uv run python scripts/data/abo_v2i_retrieval/create_data.py --out work/abo_v2i
  uv run python scripts/data/abo_v2i_retrieval/create_data.py --out work/abo_v2i \
      --repo-id mteb/ABO-V2I --push
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Image, Value, Video, load_dataset
from huggingface_hub import HfApi, create_repo

SOURCE_REPO = "hubxrt/ABO-I2V"
SOURCE_REVISION = "579f00a6d09a8b10295bc986911f832882577d62"
SPLIT = "test"

CARD = """---
license: cc-by-4.0
tags:
- video-to-image-retrieval
language:
- zxx
pretty_name: ABO-V2I
size_categories:
- 1K<n<10K
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: default
  data_files:
  - split: test
    path: data/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
---

# ABO-V2I

Video-to-image product retrieval derived from [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).
Built for [MTEB](https://github.com/embeddings-benchmark/mteb) as `ABOV2IRetrieval`.

This is the reverse direction of [`hubxrt/ABO-I2V`](https://huggingface.co/datasets/hubxrt/ABO-I2V)
(revision `{source_revision}`, task `ABOI2VRetrieval`). The media are the same files.
Only the query and corpus roles are swapped.

| config | split | rows | contents |
|---|---|---|---|
| `queries` | `test` | {n} | `_id`, `video` (360-degree turntable spin: h264, 384px long side, 24 frames, 12 fps, 2.0 s) |
| `corpus` | `test` | {n} | `_id`, `image` (catalog photograph of the product in a room or scene, 256px long side) |
| `default` | `test` | {n} | `query-id`, `corpus-id`, `score` (1:1, score 1) |

Each query is a product's turntable spin. Its single relevant document is an independently
shot catalog photograph of the same product, taken at a different time and place under
different lighting, so the target is never a frame of the query video. Product types are
CHAIR, SOFA, TABLE, HOME_FURNITURE_AND_DECOR and LAMP. See the ABO-I2V card and
`scripts/data/abo_i2v_retrieval/create_data.py` in MTEB for the selection rules.

## License and attribution

This derived dataset is distributed under the **Creative Commons Attribution 4.0 International
Public License (CC BY 4.0)**, the license of the source data:
<https://creativecommons.org/licenses/by/4.0/>

Credit for the data, including all images, must be given to:

> Amazon.com

Credit for building the source dataset, archives and benchmark sets must be given to:

> Matthieu Guillaumin (Amazon.com), Thomas Dideriksen (Amazon.com), Kenan Deng (Amazon.com),
> Himanshu Arora (Amazon.com), Arnab Dhua (Amazon.com), Xi (Brian) Zhang (Amazon.com),
> Tomas Yago-Vicente (Amazon.com), Jasmine Collins (UC Berkeley), Shubham Goel (UC Berkeley)
> and Jitendra Malik (UC Berkeley)

**Changes made to the licensed material:** turntable spin frame sequences were subsampled to 24
frames and re-encoded to H.264 video at 384px on the long side; catalog images were filtered and
subset-selected, and are redistributed at the 256px resolution published in ABO's
`images/small/`. No source pixels were otherwise altered. This repository swaps the
query and corpus roles of `hubxrt/ABO-I2V` and does not change any media.

The licensed material is provided as-is, without warranties or conditions of any kind; see
Section 5 of CC BY 4.0 for the full disclaimer of warranties and limitation of liability.

## Citation

```bibtex
@article{{collins2022abo,
  author = {{Collins, Jasmine and Goel, Shubham and Deng, Kenan and Luthra, Achleshwar and Xu, Leon and Gundogdu, Erhan and Zhang, Xi and Yago Vicente, Tomas F and Dideriksen, Thomas and Arora, Himanshu and Guillaumin, Matthieu and Malik, Jitendra}},
  journal = {{CVPR}},
  title = {{ABO: Dataset and Benchmarks for Real-World 3D Object Understanding}},
  year = {{2022}},
}}
```
"""


def _load(config: str):
    return load_dataset(SOURCE_REPO, config, split=SPLIT, revision=SOURCE_REVISION)


def build():
    src_corpus = _load("corpus").cast_column("video", Video(decode=False))
    src_queries = _load("queries").cast_column("image", Image(decode=False))
    src_qrels = _load("default")

    queries = src_corpus.cast_column("_id", Value("string"))
    corpus = src_queries.cast_column("_id", Value("string"))
    qrels = (
        src_qrels.rename_columns({"query-id": "corpus-id", "corpus-id": "query-id"})
        .select_columns(["query-id", "corpus-id", "score"])
        .cast_column("query-id", Value("string"))
        .cast_column("corpus-id", Value("string"))
        .cast_column("score", Value("int32"))
    )

    qids, cids = set(queries["_id"]), set(corpus["_id"])
    assert len(qids) == len(queries), "duplicate query _id"
    assert len(cids) == len(corpus), "duplicate corpus _id"
    assert set(qrels["query-id"]) == qids, "qrels do not cover every query video"
    assert set(qrels["corpus-id"]) <= cids, "qrels point at a missing image"
    assert len(qrels) == len(set(zip(qrels["query-id"], qrels["corpus-id"])))

    # decode=False above only avoids decoding media while swapping roles; restore
    # the decoding features so the published schema does not carry decode=False.
    queries = queries.cast_column("video", Video())
    corpus = corpus.cast_column("image", Image())

    print(f"queries={len(queries)}  corpus={len(corpus)}  qrels={len(qrels)}")
    print(f"queries features: {queries.features}")
    print(f"corpus features : {corpus.features}")
    print(f"qrels features  : {qrels.features}")
    return queries, corpus, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/abo_v2i", help="local parquet + card dir")
    ap.add_argument("--repo-id", default="mteb/ABO-V2I")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    queries, corpus, qrels = build()
    card = CARD.format(source_revision=SOURCE_REVISION, n=len(queries))

    out = Path(args.out)
    for ds, sub in ((corpus, "corpus"), (queries, "queries"), (qrels, "data")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        ds.to_parquet(out / sub / f"{SPLIT}-00000-of-00001.parquet")
    (out / "README.md").write_text(card)
    print(f"wrote local copy to {out}")

    if not args.push:
        print("--push not set; nothing uploaded.")
        return

    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    for ds, cfg in ((corpus, "corpus"), (queries, "queries"), (qrels, "default")):
        ds.push_to_hub(args.repo_id, config_name=cfg, split=SPLIT)
        print(f"pushed config {cfg!r}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"\nrevision: {api.dataset_info(args.repo_id).sha}")


if __name__ == "__main__":
    main()
