#!/usr/bin/env python3
"""Build the Multi30k image-text retrieval tasks for MTEB.

Source is `romrawinjp/multi30k` on the Hub at a pinned revision, a third-party upload of
Multi30k (Elliott et al. 2016; Barrault et al. 2018). Nothing about the captions or
images is altered here; the only work is reshaping one row per image with parallel
en/cs/de/fr captions into MTEB's standard retrieval format, so the task file needs no
custom `load_data`.

The image side is identical across the four language subsets. Rather than duplicate it
per language, the image table is written once and every `<lang>-corpus` (t2i) or
`<lang>-queries` (i2t) config points at the same file through the card's `configs:`
block.

The image table is written through the `datasets` API rather than raw pyarrow so the
parquet carries the `Image` feature in its schema metadata. Writing the struct directly
produces a plain `{bytes, path}` column that will not decode as an image on load.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/multi30k_retrieval/create_data.py --stage build

  # Build and publish both directions.
  uv run python scripts/data/multi30k_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Image
from huggingface_hub import HfApi, hf_hub_download

_SOURCE_REPO = "romrawinjp/multi30k"
_SOURCE_REV = "110e827dac7d6aabe6201d13bbdbc7413630390d"
_LANGS = ["en", "cs", "de", "fr"]
_TARGETS = {"t2i": "vnahata/Multi30k-T2I", "i2t": "vnahata/Multi30k-I2T"}
_LICENSE = "mit"


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        _SOURCE_REPO,
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=_SOURCE_REV,
    )
    tbl = pq.read_table(path)
    ids = [str(i) for i in range(tbl.num_rows)]

    # via datasets so the Image feature lands in the parquet schema metadata
    images = tbl.select(["image"]).to_pylist()
    Dataset.from_list(
        [
            {"id": f"image-{i}", "image": r["image"]}
            for i, r in zip(ids, images, strict=True)
        ]
    ).cast_column("image", Image()).to_parquet(str(work / "images.parquet"))

    counts = {"images": len(ids)}
    for lang in _LANGS:
        caps = tbl.column(lang).to_pylist()
        rows = [
            {"id": f"text-{i}-{lang}", "text": (c or "").strip()}
            for i, c in zip(ids, caps, strict=True)
            if (c or "").strip()
        ]
        pq.write_table(pa.Table.from_pylist(rows), work / f"captions_{lang}.parquet")
        kept = {r["id"].split("-")[1] for r in rows}
        for direction in ("t2i", "i2t"):
            pairs = [
                (
                    {
                        "query-id": f"text-{i}-{lang}",
                        "corpus-id": f"image-{i}",
                        "score": 1,
                    }
                    if direction == "t2i"
                    else {
                        "query-id": f"image-{i}",
                        "corpus-id": f"text-{i}-{lang}",
                        "score": 1,
                    }
                )
                for i in ids
                if i in kept
            ]
            pq.write_table(
                pa.Table.from_pylist(pairs), work / f"qrels_{direction}_{lang}.parquet"
            )
        counts[lang] = len(rows)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(direction: str) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - text-to-image" if direction == "t2i" else "  - image-to-text",
        "language:",
        *[f"  - {lg}" for lg in _LANGS],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for lg in _LANGS:
        corpus = "images" if direction == "t2i" else f"captions_{lg}"
        queries = f"captions_{lg}" if direction == "t2i" else "images"
        for suffix, src in (
            ("corpus", corpus),
            ("queries", queries),
            ("qrels", f"qrels_{direction}_{lg}"),
        ):
            lines += [
                f"  - config_name: {lg}-{suffix}",
                "    data_files:",
                "      - split: test",
                f"        path: {src}.parquet",
            ]
    lines += [
        "---",
        "",
        f"# Multi30k {direction} retrieval (MTEB)",
        "",
        "Multi30k reshaped into MTEB's standard retrieval format. The Czech, German and",
        "French captions are independent human translations describing the same Flickr30k",
        "image rather than machine translations, so the four subsets are directly comparable.",
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE.upper()}. The image",
        "side is identical across languages, so it is stored once and every language config",
        "points at the same file.",
        "",
        "Built by `scripts/data/multi30k_retrieval/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    for direction, repo in _TARGETS.items():
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        files = (
            ["images.parquet"]
            + [f"captions_{lg}.parquet" for lg in _LANGS]
            + [f"qrels_{direction}_{lg}.parquet" for lg in _LANGS]
        )
        for name in files:
            api.upload_file(
                path_or_fileobj=str(work / name),
                path_in_repo=name,
                repo_id=repo,
                repo_type="dataset",
            )
        api.upload_file(
            path_or_fileobj=io.BytesIO(_card(direction).encode()),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"pushed {repo}: {len(files)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("multi30k_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
