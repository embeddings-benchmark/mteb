"""Build the FoleyBench video/audio retrieval dataset for MTEB.

FoleyBench (Dixit et al., ICASSP 2026, arXiv:2511.13219) is a benchmark for Foley-style
video-to-audio (V2A) generation: 5,000 (video, ground-truth audio, caption) triplets of
~10 s clips from FineVideo, V3C1 and LVBench, filtered so that the sound is non-speech,
non-music and caused by something visible on screen.

The source repo ships each clip as one MP4 with its audio track. This script samples a
fixed-seed subset and splits each clip into

  * a silent video (audio track removed, re-encoded to at most 360 px high), and
  * its Foley audio track (lossless FLAC, original sample rate and channels).

One dataset serves three MTEB tasks:

  FoleyBenchV2ARetrieval   silent video          -> its audio     (the benchmark's task)
  FoleyBenchVT2ARetrieval  silent video + caption -> its audio    (text-conditioned V2A)
  FoleyBenchA2VRetrieval   audio                 -> its silent video (roles swapped in
                           ``dataset_transform``)

Configs (split ``test``):

  queries   ``id`` (``video-<key>``), ``video``, ``text`` (the FoleyBench caption), and
            provenance: ``source_dataset``, ``source_metadata`` (YouTube / Vimeo id and
            start/end time in the original video).
  corpus    ``id`` (``audio-<key>``), ``audio``.
  default   qrels, ``query-id`` -> ``corpus-id``, score 1 (one audio per video).

Selection: the 5,000 keys are shuffled with a fixed seed, and the first ``--n`` clips
that have both a video and an audio stream, and whose MP4 bytes and source segment are
not already selected, are kept.

FoleyBench is CC BY-NC-SA 4.0 and its terms require anyone re-hosting it to include the
terms and make users agree to them, so the MTEB copy is gated with the same terms.

Usage:
  python scripts/data/foleybench_retrieval/create_data.py --src work/foleybench --work work/fb
  python scripts/data/foleybench_retrieval/create_data.py --src work/foleybench --work work/fb --push
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Value, Video
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
from tqdm import tqdm

SRC_REPO = "FoleyBench/foleybench"
SRC_REVISION = "f4be31fd6ddb715a17c9220e4f9ea6ac5cf91d23"
SPLIT = "test"
SEED = 42
MAX_HEIGHT = 360


def fetch_source(src: Path) -> Path:
    snapshot_download(
        SRC_REPO, repo_type="dataset", revision=SRC_REVISION, local_dir=src
    )
    return src


def read_gated_terms() -> str:
    card = Path(
        hf_hub_download(
            SRC_REPO, "README.md", repo_type="dataset", revision=SRC_REVISION
        )
    ).read_text()
    front = card.split("---", 2)[1]
    start = front.index("extra_gated_prompt:")
    end = front.index("\ndataset_info:", start)
    return front[start:end].rstrip() + "\n"


def iter_row_batches(src: Path, keys: set[int]):
    """Yield, per parquet row group (50 rows), the source rows whose key is in ``keys``.

    Rows carry the base64 MP4, so reading a row group at a time keeps memory bounded.
    """
    for shard in sorted((src / "data").glob("train-*.parquet")):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            mask = [k in keys for k in table.column("key").to_pylist()]
            rows = table.filter(mask).to_pylist()
            if rows:
                yield rows


def _ffprobe_streams(path: Path) -> set[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout  # fmt: skip
    return {line.strip() for line in out.splitlines() if line.strip()}


def split_clip(row: dict, out: Path) -> dict | None:
    """Write the silent video and the audio track of one clip; None if unusable."""
    key = row["key"]
    mp4 = base64.b64decode(row["video_data"])
    video_out = out / "video" / f"{key}.mp4"
    audio_out = out / "audio" / f"{key}.flac"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.mp4"
        src.write_bytes(mp4)
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
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_out)],
            check=True, timeout=600,
        )  # fmt: skip
        subprocess.run(
            common + ["-vn", "-sn", "-dn", "-map_metadata", "-1",
                      "-map", "0:a:0", "-c:a", "flac", str(audio_out)],
            check=True, timeout=600,
        )  # fmt: skip
    assert "audio" not in _ffprobe_streams(video_out), video_out
    meta = ast.literal_eval(row["metadata"])
    return {
        "key": key,
        "sha": hashlib.sha256(mp4).hexdigest(),
        "segment": (
            meta.get("yt-id") or meta.get("vimeo-id"),
            meta.get("start_time"),
            meta.get("end_time"),
        ),
        "caption": row["caption"],
        "source_dataset": row["dataset"],
        "source_metadata": json.dumps(meta, sort_keys=True),
        "video": str(video_out),
        "audio": str(audio_out),
    }


def select(src: Path, work: Path, n: int, workers: int) -> list[dict]:
    meta = pd.read_csv(src / "foleybench.csv")
    order = np.random.default_rng(SEED).permutation(meta["key"].to_numpy()).tolist()
    candidates = order[: int(n * 1.05)]
    out = work / "media"
    for sub in ("video", "audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    done: dict[int, dict | None] = {}
    with (
        ThreadPoolExecutor(workers) as ex,
        tqdm(total=len(candidates), desc="split clips") as bar,
    ):
        for rows in iter_row_batches(src, set(candidates)):
            for row, item in zip(rows, ex.map(lambda r: split_clip(r, out), rows)):
                done[row["key"]] = item
                bar.update()
    kept, shas, segments = [], set(), set()
    for key in candidates:
        item = done.get(key)
        if item is None or item["sha"] in shas or item["segment"] in segments:
            print(f"skip key {key}")
            continue
        shas.add(item["sha"])
        segments.add(item["segment"])
        kept.append(item)
        if len(kept) == n:
            return kept
    raise RuntimeError(f"only {len(kept)} usable clips, wanted {n}")


CARD = """---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: FoleyBench (MTEB retrieval subset)
size_categories:
- n<1K
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

# FoleyBench (MTEB retrieval subset)

Video/audio retrieval built from [FoleyBench](https://huggingface.co/datasets/FoleyBench/foleybench)
(revision `{src_rev}`) for [MTEB](https://github.com/embeddings-benchmark/mteb).
FoleyBench is a benchmark for Foley-style video-to-audio generation: each clip has a
visible sound source and non-speech, non-music audio caused by on-screen events.

{n} clips were sampled from the 5,000 with a fixed seed. Each clip is split into a silent
video and its audio track.

| config | split | rows | contents |
|---|---|---|---|
| `queries` | `test` | {n} | `id`, `video` (silent, at most 360 px high), `text` (FoleyBench caption), `source_dataset`, `source_metadata` |
| `corpus` | `test` | {n} | `id`, `audio` (the clip's audio track, lossless FLAC) |
| `default` | `test` | {n} | `query-id`, `corpus-id`, `score` (1) |

Used by `FoleyBenchV2ARetrieval`, `FoleyBenchVT2ARetrieval` (video + caption) and
`FoleyBenchA2VRetrieval` (roles swapped). Build script:
`scripts/data/foleybench_retrieval/create_data.py` in MTEB.

## License and terms

CC BY-NC-SA 4.0, following FoleyBench, for academic research only. Each video keeps the
license of its original source; `source_dataset` and `source_metadata` give its
provenance (YouTube or Vimeo id and time range). The FoleyBench terms of use, which this
repo is gated with, apply in full. Videos removed from FoleyBench on request will also be
removed here. Changes: subset selection; audio removed from the videos and stored
separately; videos re-encoded to at most 360 px high.

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


def _write(ds: Dataset, d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    embedded = ds.with_format("arrow").map(embed_table_storage, batched=True)
    embedded.with_format(None).to_parquet(d / f"{SPLIT}-00000-of-00001.parquet")


def package(items: list[dict], out: Path) -> None:
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
    _write(queries, out / "queries")
    _write(corpus, out / "corpus")
    _write(qrels, out / "data")
    card = CARD.format(
        gated_terms=read_gated_terms(), src_rev=SRC_REVISION, n=len(items)
    )
    (out / "README.md").write_text(card)
    print(f"wrote {out}: {len(items)} clips")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="work/foleybench_src")
    ap.add_argument("--work", default="work/foleybench_retrieval")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repo-id", default="mteb/FoleyBench")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    src = fetch_source(Path(args.src))
    items = select(src, work, args.n, args.workers)
    print(
        "sources:",
        pd.Series([it["source_dataset"] for it in items]).value_counts().to_dict(),
    )
    out = work / "hf"
    package(items, out)

    if args.push:
        # Gated with automatic approval: the FoleyBench terms require users of any
        # re-hosted copy to agree to them (the prompt is in the card's front matter).
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.update_repo_settings(args.repo_id, repo_type="dataset", gated="auto")
        api.upload_large_folder(
            folder_path=str(out), repo_id=args.repo_id, repo_type="dataset"
        )
        print(f"{args.repo_id} revision: {api.dataset_info(args.repo_id).sha}")


if __name__ == "__main__":
    main()
