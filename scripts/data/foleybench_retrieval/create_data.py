"""Build the FoleyBench and FoleyBench-Long video/audio retrieval datasets for MTEB.

FoleyBench (Dixit et al., ICASSP 2026, arXiv:2511.13219) is a benchmark for Foley-style
video-to-audio (V2A) generation: (video, ground-truth audio, caption) triplets filtered so
that the sound is non-speech, non-music and caused by something visible on screen.

  base   FoleyBench/foleybench       5,000 ~10 s clips (FineVideo, V3C1, LVBench)
  long   FoleyBench/foleybench-long    653  30 s clips (FineVideo, V3C1)

The source repos ship each clip as one MP4 with its audio track. This script keeps every
clip that has both a video and an audio stream, as in the source, and splits each into

  * a silent video (audio track removed, re-encoded to at most 360 px high), and
  * its Foley audio track (lossless FLAC, original sample rate and channels).

Each dataset serves three MTEB tasks (``FoleyBench*`` for base, ``FoleyBenchLong*`` for
long):

  ...V2ARetrieval   silent video           -> its audio     (the benchmark's task)
  ...VT2ARetrieval  silent video + caption -> its audio     (text-conditioned V2A)
  ...A2VRetrieval   audio                  -> its silent video (roles swapped in
                    ``dataset_transform``)

Configs (split ``test``):

  queries   ``id`` (``video-<key>``), ``video``, ``text`` (the FoleyBench caption), and
            provenance: ``source_dataset``, ``source_metadata`` (YouTube / Vimeo / V3C id
            and start/end time in the original video).
  corpus    ``id`` (``audio-<key>``), ``audio``.
  default   qrels, ``query-id`` -> ``corpus-id``, score 1 (one audio per video).

FoleyBench is CC BY-NC-SA 4.0 and its terms require anyone re-hosting it to include the
terms and make users agree to them, so the MTEB copies are gated with the same terms.

Usage:
  python scripts/data/foleybench_retrieval/create_data.py --variant base --work work/fb
  python scripts/data/foleybench_retrieval/create_data.py --variant long --work work/fbl --push
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Value, Video
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
from tqdm import tqdm

VARIANTS = {
    "base": {
        "src_repo": "FoleyBench/foleybench",
        "src_revision": "f4be31fd6ddb715a17c9220e4f9ea6ac5cf91d23",
        "repo_id": "mteb/FoleyBench",
        "name": "FoleyBench",
        "task_prefix": "FoleyBench",
        "clips": "~10 s clips from FineVideo, V3C1 and LVBench",
    },
    "long": {
        "src_repo": "FoleyBench/foleybench-long",
        "src_revision": "7640c21a44ba78d88621847383aaceb138f51300",
        "repo_id": "mteb/FoleyBenchLong",
        "name": "FoleyBench-Long",
        "task_prefix": "FoleyBenchLong",
        "clips": "30 s clips from FineVideo and V3C1",
    },
}
SPLIT = "test"
MAX_HEIGHT = 360
SHARD_BYTES = 1_000_000_000


def fetch_source(cfg: dict, src: Path) -> Path:
    snapshot_download(
        cfg["src_repo"],
        repo_type="dataset",
        revision=cfg["src_revision"],
        local_dir=src,
        allow_patterns=["data/*.parquet", "README.md"],
    )
    return src


def read_gated_terms(cfg: dict) -> str:
    card = Path(
        hf_hub_download(
            cfg["src_repo"],
            "README.md",
            repo_type="dataset",
            revision=cfg["src_revision"],
        )
    ).read_text()
    front = card.split("---", 2)[1]
    start = front.index("extra_gated_prompt:")
    end = front.index("\ndataset_info:", start)
    return front[start:end].rstrip() + "\n"


def iter_row_batches(src: Path):
    """Yield the source rows one parquet row group at a time.

    Rows carry the base64 MP4, so reading a row group at a time keeps memory bounded.
    """
    for shard in sorted((src / "data").glob("train-*.parquet")):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            yield pf.read_row_group(rg).to_pylist()


def _ffprobe_streams(path: Path) -> set[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout  # fmt: skip
    # Some streams print extra empty fields (e.g. "video,"), so keep the first field only.
    return {line.split(",")[0].strip() for line in out.splitlines() if line.strip()}


def split_clip(row: dict, out: Path) -> dict | None:
    """Write the silent video and the audio track of one clip; None if unusable.

    Outputs are written to a temporary dir and moved into place, so a rerun reuses the
    clips that an interrupted run already finished.
    """
    key = row["key"]
    video_out = out / "video" / f"{key}.mp4"
    audio_out = out / "audio" / f"{key}.flac"
    if not (video_out.exists() and audio_out.exists()):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.mp4"
            src.write_bytes(base64.b64decode(row["video_data"]))
            try:
                streams = _ffprobe_streams(src)
            except subprocess.CalledProcessError:
                return None
            if not {"video", "audio"} <= streams:
                return None
            common = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src)]
            subprocess.run(
                common + ["-an", "-sn", "-dn", "-map_metadata", "-1",
                          "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                          f"{tmp}/video.mp4"],
                check=True, timeout=600,
            )  # fmt: skip
            subprocess.run(
                common + ["-vn", "-sn", "-dn", "-map_metadata", "-1",
                          "-map", "0:a:0", "-c:a", "flac", f"{tmp}/audio.flac"],
                check=True, timeout=600,
            )  # fmt: skip
            assert "audio" not in _ffprobe_streams(Path(tmp) / "video.mp4"), key
            shutil.move(f"{tmp}/audio.flac", audio_out)
            shutil.move(f"{tmp}/video.mp4", video_out)
    meta = ast.literal_eval(row["metadata"])
    return {
        "key": key,
        "caption": row["caption"],
        "source_dataset": row["dataset"],
        "source_metadata": json.dumps(meta, sort_keys=True),
        "video": str(video_out),
        "audio": str(audio_out),
    }


def split_all(src: Path, work: Path, workers: int) -> tuple[list[dict], int]:
    """Split every source clip; return the ones with video and audio, in key order."""
    out = work / "media"
    for sub in ("video", "audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    done: dict[int, dict | None] = {}
    with ThreadPoolExecutor(workers) as ex, tqdm(desc="split clips") as bar:
        for rows in iter_row_batches(src):
            for row, item in zip(rows, ex.map(lambda r: split_clip(r, out), rows)):
                done[row["key"]] = item
                bar.update()
    kept = []
    for key in sorted(done):
        if done[key] is None:
            print(f"skip key {key}: no video or no audio stream")
        else:
            kept.append(done[key])
    return kept, len(done)


CARD = """---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: {name} (MTEB retrieval)
size_categories:
- {size_category}
task_categories:
- any-to-any
tags:
- video-to-audio
- foley
{gated_terms}configs:
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

# {name} (MTEB retrieval)

Video/audio retrieval built from [{name}](https://huggingface.co/datasets/{src_repo})
(revision `{src_rev}`) for [MTEB](https://github.com/embeddings-benchmark/mteb).
FoleyBench is a benchmark for Foley-style video-to-audio generation: each clip has a
visible sound source and non-speech, non-music audio caused by on-screen events.

{name} has {n_src:,} {clips}. Every clip with both a video and an audio track is used
({n:,}), as in the source, and each is split into a silent video and its audio track.

| config | split | rows | contents |
|---|---|---|---|
| `queries` | `test` | {n:,} | `id`, `video` (silent, at most 360 px high), `text` (FoleyBench caption), `source_dataset`, `source_metadata` |
| `corpus` | `test` | {n:,} | `id`, `audio` (the clip's audio track, lossless FLAC) |
| `default` | `test` | {n:,} | `query-id`, `corpus-id`, `score` (1) |

Used by `{prefix}V2ARetrieval`, `{prefix}VT2ARetrieval` (video + caption) and
`{prefix}A2VRetrieval` (roles swapped). Build script:
`scripts/data/foleybench_retrieval/create_data.py` in MTEB.

## License and terms

CC BY-NC-SA 4.0, following FoleyBench, for academic research only. Each video keeps the
license of its original source; `source_dataset` and `source_metadata` give its
provenance (YouTube, Vimeo or V3C id and time range). The {name} terms of use, which this
repo is gated with, apply in full. Videos removed from {name} on request will also be
removed here. Changes: audio removed from the videos and stored separately; videos
re-encoded to at most 360 px high.

## Citation

```bibtex
@inproceedings{{dixit2026foleybench,
  author = {{Dixit, Satvik and Saito, Koichi and Zhong, Zhi and Mitsufuji, Yuki and Donahue, Chris}},
  booktitle = {{ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
  title = {{FoleyBench: A Benchmark For Video-to-Audio Models}},
  year = {{2026}},
}}
```
"""


def _write(ds: Dataset, d: Path, media_bytes: int = 0) -> None:
    """Write ``ds`` with its media embedded, in parquet shards of about 1 GB."""
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    num_shards = max(1, -(-media_bytes // SHARD_BYTES))
    for i in range(num_shards):
        shard = ds.shard(num_shards, i, contiguous=True)
        embedded = shard.with_format("arrow").map(embed_table_storage, batched=True)
        embedded.with_format(None).to_parquet(
            d / f"{SPLIT}-{i:05d}-of-{num_shards:05d}.parquet"
        )


def package(cfg: dict, items: list[dict], n_src: int, out: Path) -> None:
    queries = (
        Dataset.from_dict(
            {
                "id": [f"video-{it['key']}" for it in items],
                "video": [it["video"] for it in items],
                "text": [it["caption"] for it in items],
                "source_dataset": [it["source_dataset"] for it in items],
                "source_metadata": [it["source_metadata"] for it in items],
            }
        )
        .cast_column("video", Video())
        .cast_column("id", Value("string"))
    )
    corpus = Dataset.from_dict(
        {
            "id": [f"audio-{it['key']}" for it in items],
            "audio": [it["audio"] for it in items],
        }
    ).cast_column("audio", Audio())
    qrels = Dataset.from_dict(
        {
            "query-id": [f"video-{it['key']}" for it in items],
            "corpus-id": [f"audio-{it['key']}" for it in items],
            "score": [1] * len(items),
        }
    ).cast_column("score", Value("int32"))
    video_bytes = sum(Path(it["video"]).stat().st_size for it in items)
    audio_bytes = sum(Path(it["audio"]).stat().st_size for it in items)
    _write(queries, out / "queries", video_bytes)
    _write(corpus, out / "corpus", audio_bytes)
    _write(qrels, out / "data")
    card = CARD.format(
        name=cfg["name"],
        size_category="n<1K" if len(items) < 1000 else "1K<n<10K",
        gated_terms=read_gated_terms(cfg),
        src_repo=cfg["src_repo"],
        src_rev=cfg["src_revision"],
        n_src=n_src,
        clips=cfg["clips"],
        n=len(items),
        prefix=cfg["task_prefix"],
    )
    (out / "README.md").write_text(card)
    print(f"wrote {out}: {len(items)} of {n_src} clips")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    ap.add_argument("--src", help="source download dir (default: <work>/src)")
    ap.add_argument("--work", help="work dir (default: work/foleybench_<variant>)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    work = Path(args.work or f"work/foleybench_{args.variant}")
    src = fetch_source(cfg, Path(args.src) if args.src else work / "src")
    items, n_src = split_all(src, work, args.workers)
    print(
        "sources:",
        pd.Series([it["source_dataset"] for it in items]).value_counts().to_dict(),
    )
    out = work / "hf"
    package(cfg, items, n_src, out)

    if args.push:
        # Gated with automatic approval: the FoleyBench terms require users of any
        # re-hosted copy to agree to them (the prompt is in the card's front matter).
        repo_id = cfg["repo_id"]
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.update_repo_settings(repo_id, repo_type="dataset", gated="auto")
        api.upload_large_folder(
            folder_path=str(out), repo_id=repo_id, repo_type="dataset"
        )
        # Shards from an earlier build would still match the ``test-*`` globs.
        local = {p.relative_to(out).as_posix() for p in out.glob("*/*.parquet")}
        stale = [
            f
            for f in api.list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".parquet") and f not in local
        ]
        if stale:
            api.delete_files(
                repo_id,
                stale,
                repo_type="dataset",
                commit_message="Remove shards from the previous build",
            )
        print(f"{repo_id} revision: {api.dataset_info(repo_id).sha}")


if __name__ == "__main__":
    main()
