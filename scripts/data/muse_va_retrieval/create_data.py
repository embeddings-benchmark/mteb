#!/usr/bin/env python3
"""Package jiahaomei/MUSE-VA into MTEB audio↔image retrieval format.

MUSE-VA pairs music clips with emotion-aligned generated images (~625 test pairs).
Each audio clip has exactly one relevant image and vice versa.

Builds two Hub datasets (corpus / queries / qrels configs, test split):
  - {repo-prefix}-A2I  (audio query → image corpus)
  - {repo-prefix}-I2A  (image query → audio corpus)

Usage:
  export HF_TOKEN=...
  export HF_HUB_ENABLE_HF_TRANSFER=1   # optional
  uv run python scripts/data/muse_va_retrieval/create_data.py \\
      --repo-prefix Wissam42/MUSE-VA \\
      --work-dir /tmp/muse_va_mteb \\
      --push
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Dataset, DatasetDict, Image
from huggingface_hub import create_repo, snapshot_download
from PIL import Image as PILImage
from tqdm import tqdm

_SOURCE = "jiahaomei/MUSE-VA"


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


def _audio_ext(path_hint: str | None) -> str:
    if path_hint:
        suf = Path(path_hint).suffix.lower()
        if suf in {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}:
            return suf
    return ".wav"


def _save_resized_jpeg(raw: bytes, dst: Path, max_side: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize(
                (int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS
            )
        im.save(dst, format="JPEG", quality=90)


def _extract_struct_bytes(cell) -> tuple[bytes, str | None]:
    """Pull ``(bytes, path_hint)`` from a datasets/HF audio or image struct."""
    if cell is None:
        raise ValueError("empty media cell")
    # pyarrow may return dict-like or object with as_py()
    if hasattr(cell, "as_py"):
        cell = cell.as_py()
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        path_hint = cell.get("path")
        if raw is None and path_hint and Path(path_hint).is_file():
            return Path(path_hint).read_bytes(), path_hint
        if raw is None:
            raise ValueError(f"media struct missing bytes: keys={list(cell)}")
        return raw, path_hint
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell), None
    raise TypeError(f"unsupported media cell type: {type(cell)!r}")


def _extract_from_parquets(
    parquet_files: list[Path],
    *,
    audio_dir: Path,
    image_dir: Path,
    max_side: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    audio_ids: list[str] = []
    image_ids: list[str] = []
    audio_paths: list[str] = []
    image_paths: list[str] = []

    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    for pq_path in tqdm(parquet_files, desc="parquet shards"):
        table = pq.read_table(pq_path, columns=["id", "audio", "image"])
        ids = table.column("id")
        audios = table.column("audio")
        images = table.column("image")
        for i in range(table.num_rows):
            rid = ids[i].as_py()
            aid = f"aud-{rid}"
            iid = f"img-{rid}"

            audio_bytes, audio_hint = _extract_struct_bytes(audios[i])
            image_bytes, _ = _extract_struct_bytes(images[i])

            ap = audio_dir / f"{aid}{_audio_ext(audio_hint)}"
            ip = image_dir / f"{iid}.jpg"
            if not ap.exists():
                ap.write_bytes(audio_bytes)
            _save_resized_jpeg(image_bytes, ip, max_side)

            audio_ids.append(aid)
            image_ids.append(iid)
            audio_paths.append(str(ap))
            image_paths.append(str(ip))

        # Drop shard from RAM before opening the next one.
        del table

    return audio_ids, image_ids, audio_paths, image_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-prefix", default="Wissam42/MUSE-VA")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/muse_va_mteb"))
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already-downloaded test shards under --work-dir/source",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    source_dir = work / "source"
    data_dir = source_dir / "data"

    if not args.skip_download:
        print(
            f"Downloading {_SOURCE} *test* shards only "
            f"(~21 GB across 106 files; train/valid ~185 GB are skipped)…"
        )
        snapshot_download(
            _SOURCE,
            repo_type="dataset",
            local_dir=source_dir,
            allow_patterns=["data/test-*", "README.md", ".gitattributes"],
        )
    elif not any(data_dir.glob("test-*.parquet")):
        raise SystemExit(f"--skip-download set but no test shards under {data_dir}/")

    parquet_files = sorted(data_dir.glob("test-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No test-*.parquet under {data_dir}")
    print(f"Found {len(parquet_files)} test shards")

    audio_ids, image_ids, audio_paths, image_paths = _extract_from_parquets(
        parquet_files,
        audio_dir=work / "media" / "audio",
        image_dir=work / "media" / "image",
        max_side=args.max_side,
    )
    n = len(audio_ids)
    print(f"Extracted {n} pairs → {work / 'media'}")

    a2i_corpus = Dataset.from_dict({"id": image_ids, "image": image_paths}).cast_column(
        "image", Image()
    )
    a2i_queries = Dataset.from_dict(
        {"id": audio_ids, "audio": audio_paths}
    ).cast_column("audio", Audio())
    a2i_qrels = Dataset.from_dict(
        {
            "query-id": audio_ids,
            "corpus-id": image_ids,
            "score": [1] * n,
        }
    )

    i2a_corpus = Dataset.from_dict({"id": audio_ids, "audio": audio_paths}).cast_column(
        "audio", Audio()
    )
    i2a_queries = Dataset.from_dict(
        {"id": image_ids, "image": image_paths}
    ).cast_column("image", Image())
    i2a_qrels = Dataset.from_dict(
        {
            "query-id": image_ids,
            "corpus-id": audio_ids,
            "score": [1] * n,
        }
    )

    print(
        f"A2I corpus={len(a2i_corpus)} queries={len(a2i_queries)} "
        f"qrels={len(a2i_qrels)}"
    )
    print(
        f"I2A corpus={len(i2a_corpus)} queries={len(i2a_queries)} "
        f"qrels={len(i2a_qrels)}"
    )

    token = os.environ.get("HF_TOKEN") if args.push else None
    if args.push and not token:
        raise SystemExit("Set HF_TOKEN to push")

    out = None if args.push else work / "mteb_export"
    _push_retrieval_direction(
        repo_id=f"{args.repo_prefix}-A2I",
        queries=a2i_queries,
        corpus=a2i_corpus,
        qrels=a2i_qrels,
        token=token,
        out_dir=out,
    )
    _push_retrieval_direction(
        repo_id=f"{args.repo_prefix}-I2A",
        queries=i2a_queries,
        corpus=i2a_corpus,
        qrels=i2a_qrels,
        token=token,
        out_dir=out,
    )


if __name__ == "__main__":
    main()
