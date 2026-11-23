"""Build the MultiVENT 2.0 text-to-video retrieval dataset for MTEB.

MultiVENT 2.0 (Kriz et al., arXiv:2410.11619) is an event-centric news video retrieval
benchmark: English queries about specific world events, and a multilingual collection
of news videos (Arabic, Chinese, English, Korean, Russian, Spanish) whose relevance often
depends on what is said or written on screen. The test collection has 109,800 videos
(about 970 GB in 1,098 tars), 2,546 queries and graded relevance judgments (0-3).

The full collection is far too large for MTEB, so this script keeps

  * all 2,546 official test queries and their official judgments, and
  * a corpus made of the judged pool: every video judged for any test query (4,222
    videos). This includes the 2,737 videos judged not relevant, which were pooled from
    retrieval runs and are therefore hard negatives. Unjudged videos are left out, so
    every corpus video has a known label for the queries it was judged against.

Only the needed videos are downloaded. Each tar holds 100 consecutive doc ids
(``doc_id // 100 + 1``), with ``<id>.csv``, ``<id>.mp4`` (silent 640x360 H.264) and
``<id>.m4a`` (AAC) per video. The script walks the tar headers with HTTP range requests
and fetches only the mp4 and m4a members it needs. Their bytes are kept unchanged.
Some videos have no m4a in the source (no sound); they get a silent 16 kHz mono AAC
track of the same length and ``has_audio`` set to false. A few mp4s also carry an
audio track; it is dropped (video stream copied) so that every video is silent.

One dataset serves two MTEB tasks:

  MultiVENT2T2VRetrieval   query -> silent video
  MultiVENT2T2VARetrieval  query -> video and its audio track

Configs (split ``test``):

  queries   ``id`` (official query id), ``text``.
  corpus    ``id`` (official doc id), ``video``, ``audio``, and provenance:
            ``video_language``, ``youtube_id``, ``has_audio``.
  default   qrels, ``query-id`` -> ``corpus-id``, ``score`` 1-3 (judgments with
            relevance 0 are dropped from the qrels; those videos stay in the corpus).

Usage:
  python scripts/data/multivent2_retrieval/create_data.py --work work/multivent2
  python scripts/data/multivent2_retrieval/create_data.py --work work/multivent2 --push
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import Audio, Dataset, Value, Video
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo, get_token, hf_hub_download, hf_hub_url
from tqdm import tqdm

SRC_REPO = "hltcoe/MultiVENT2.0"
SRC_REVISION = "60f3e8cac34207c3695dc1d673ff196daa12e9e4"
SPLIT = "test"
VIDEOS_PER_TAR = 100
CHUNK = 64 * 1024
CORPUS_SHARD_BYTES = 1_000_000_000
NO_AUDIO = ".noaudio"  # marker for videos the source ships without an audio file

CARD = """---
license: apache-2.0
task_categories:
- text-to-video
language:
- en
- ar
- zh
- ko
- ru
- es
tags:
- mteb
- video
- retrieval
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*.parquet
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*.parquet
- config_name: default
  data_files:
  - split: test
    path: data/test-*.parquet
---

# MultiVENT 2.0 (MTEB judged-pool subset)

A subset of the test set of [MultiVENT 2.0]({src_url}) (revision `{src_rev}`), an
event-centric news video retrieval benchmark, re-packaged for
[MTEB](https://github.com/embeddings-benchmark/mteb).

- **queries:** all {n_queries:,} official test queries (English).
- **corpus:** the {n_docs:,} videos judged for any test query, out of 109,800. This
  includes the videos judged not relevant, which were pooled from retrieval runs.
  Unjudged videos are left out.
- **qrels:** the official graded judgments with relevance 1-3.

Each corpus row has the original silent `video` (mp4) and `audio` (m4a) files,
byte-for-byte, plus the video's language and YouTube id. Videos that the source ships
without an audio file have a silent track of the same length and `has_audio: false`. The build script is
`scripts/data/multivent2_retrieval/create_data.py` in the MTEB repository.

Scores are not comparable with the official full-collection numbers, because the corpus
is much smaller.

## Citation

```bibtex
@misc{{kriz2025multivent20massivemultilingual,
  author = {{Kriz, Reno and Sanders, Kate and Etter, David and Murray, Kenton and Carpenter, Cameron and Van Ochten, Kelly and Recknor, Hannah and Guallar-Blasco, Jimena and Martin, Alexander and Colaianni, Ronald and King, Nolan and Yang, Eugene and Van Durme, Benjamin}},
  title = {{MultiVENT 2.0: A Massive Multilingual Benchmark for Event-Centric Video Retrieval}},
  year = {{2025}},
  eprint = {{2410.11619}},
  archivePrefix = {{arXiv}},
}}
```
"""


def read_metadata(work: Path) -> tuple[list[dict], list[dict]]:
    meta = work / "meta"
    paths = {
        name: hf_hub_download(
            SRC_REPO,
            name,
            repo_type="dataset",
            revision=SRC_REVISION,
            local_dir=meta,
        )
        for name in ["multivent_2_test_queries.csv", "multivent_2_test_judgments.jsonl"]
    }
    with open(paths["multivent_2_test_queries.csv"], newline="") as f:
        queries = [{"id": r["Query_id"], "text": r["query"]} for r in csv.DictReader(f)]
    with open(paths["multivent_2_test_judgments.jsonl"]) as f:
        judgments = [json.loads(line) for line in f]
    return queries, judgments


class TarReader:
    """Range-reads members of one remote tar without downloading the whole file."""

    def __init__(self, tar_name: str, session: requests.Session):
        self.tar_name = tar_name
        self.session = session
        self.url = ""
        self.buf_start = 0
        self.buf = b""

    def _resolve(self) -> None:
        # Follow the Hub redirect once; range requests then go straight to the CDN.
        resolve_url = hf_hub_url(
            SRC_REPO, self.tar_name, repo_type="dataset", revision=SRC_REVISION
        )
        r = self.session.head(
            resolve_url,
            headers={"Authorization": f"Bearer {get_token()}"},
            allow_redirects=False,
            timeout=60,
        )
        r.raise_for_status()
        self.url = r.headers.get("Location", resolve_url)

    def _get(self, start: int, end: int, stream: bool = False) -> requests.Response:
        for attempt in range(5):
            if not self.url:
                self._resolve()
            try:
                headers = {"Range": f"bytes={start}-{end}"}
                if self.url.startswith("https://huggingface.co"):
                    headers["Authorization"] = f"Bearer {get_token()}"
                r = self.session.get(
                    self.url, headers=headers, timeout=120, stream=stream
                )
                if r.status_code == 206:
                    if not stream:
                        r.content  # read the body here so dropped connections retry
                    return r
                if r.status_code in (401, 403):  # signed URL expired
                    self.url = ""
                r.close()
            except requests.RequestException:
                pass
        raise RuntimeError(f"range request failed: {self.tar_name} {start}-{end}")

    def header(self, off: int) -> tuple[str, int, bytes] | None:
        if not (self.buf_start <= off and off + 512 <= self.buf_start + len(self.buf)):
            self.buf_start, self.buf = off, self._get(off, off + CHUNK - 1).content
        block = self.buf[off - self.buf_start : off - self.buf_start + 512]
        if len(block) < 512 or not block.strip(b"\0"):
            return None
        name = block[:100].rstrip(b"\0").decode()
        size = int(block[124:136].rstrip(b"\0 ") or b"0", 8)
        return name, size, block[156:157]

    def save(self, off: int, size: int, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".part")
        for _ in range(5):
            try:
                with (
                    self._get(off, off + size - 1, stream=True) as r,
                    open(tmp, "wb") as f,
                ):
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            except requests.RequestException:
                continue
            if tmp.stat().st_size == size:
                tmp.rename(path)
                return
        raise RuntimeError(f"download failed for {path.name}")


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _mark_no_audio(files: Path, doc_id: str) -> None:
    (files / f"{doc_id}{NO_AUDIO}").touch()


def fetch_tar(tar_index: int, doc_ids: set[str], files: Path) -> None:
    """Fetch the mp4 and m4a of ``doc_ids`` from one tar.

    Members are stored per video in id order (csv, mp4, m4a). Some videos have no m4a;
    once the walk reaches a later video, a still-missing m4a is recorded as absent.
    """
    missing = {f"{d}.mp4" for d in doc_ids if not (files / f"{d}.mp4").exists()}
    missing |= {
        f"{d}.m4a"
        for d in doc_ids
        if not (files / f"{d}.m4a").exists() and not (files / f"{d}{NO_AUDIO}").exists()
    }
    if not missing:
        return
    reader = TarReader(f"{SPLIT}/{tar_index:06d}.tar", _session())
    off = 0
    while missing:
        entry = reader.header(off)
        if entry is None:
            break
        name, size, typeflag = entry
        member = name.rsplit("/", 1)[-1]
        stem = member.split(".", 1)[0]
        if stem.isdigit():
            for m in [m for m in missing if m.endswith(".m4a")]:
                if int(m.split(".")[0]) < int(stem):
                    _mark_no_audio(files, m.split(".")[0])
                    missing.discard(m)
        if typeflag in (b"0", b"\0") and member in missing:
            reader.save(off + 512, size, files / member)
            missing.discard(member)
        off += 512 + (size + 511) // 512 * 512
    for m in [m for m in missing if m.endswith(".m4a")]:
        _mark_no_audio(files, m.split(".")[0])
        missing.discard(m)
    if missing:
        raise RuntimeError(f"{reader.tar_name}: not found {sorted(missing)}")


def fill_silent_audio(doc_ids: list[str], files: Path, workers: int) -> None:
    """Give videos without an audio file a silent track of the same length."""

    def fill(d: str) -> None:
        out = files / f"{d}.m4a"
        if not (files / f"{d}{NO_AUDIO}").exists() or out.exists():
            return
        duration = _duration(files / f"{d}.mp4", "v")
        tmp = files / f"{d}.silent.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi"]
            + ["-i", "anullsrc=r=16000:cl=mono", "-t", f"{duration:.3f}"]
            + ["-c:a", "aac", str(tmp)],
            check=True,
        )
        tmp.rename(out)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fill, doc_ids))


def download(doc_ids: list[str], files: Path, workers: int) -> None:
    files.mkdir(parents=True, exist_ok=True)
    by_tar: dict[int, set[str]] = defaultdict(set)
    for d in doc_ids:
        by_tar[int(d) // VIDEOS_PER_TAR + 1].add(d)
    errors = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_tar, t, ds, files) for t, ds in by_tar.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="tars"):
            if fut.exception() is not None:
                errors.append(fut.exception())
    if errors:
        raise RuntimeError(f"{len(errors)} tars failed (rerun to resume): {errors[:3]}")


def _duration(path: Path, stream: str) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream]
        + ["-show_entries", "stream=codec_type:format=duration", "-of", "json"]
        + [str(path)],
        capture_output=True,
        text=True,
    )
    info = json.loads(out.stdout or "{}")
    if not info.get("streams"):
        return None
    return float(info["format"]["duration"])


def strip_video_audio(doc_ids: list[str], files: Path, workers: int) -> None:
    """Drop the audio track from the few mp4s that have one (video stream copied)."""

    def strip(d: str) -> None:
        mp4 = files / f"{d}.mp4"
        if _duration(mp4, "a") is None:
            return
        tmp = files / f"{d}.silent.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(mp4)]
            + ["-map", "0:v:0", "-c", "copy", "-an", str(tmp)],
            check=True,
        )
        tmp.rename(mp4)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(strip, doc_ids))


def verify(doc_ids: list[str], files: Path, workers: int) -> None:
    """Check that every mp4 has video and no audio, and every m4a has audio."""

    def check(d: str) -> str | None:
        mp4, m4a = files / f"{d}.mp4", files / f"{d}.m4a"
        if _duration(mp4, "v") is None or _duration(mp4, "a") is not None:
            return f"{d}: bad video"
        if _duration(m4a, "a") is None:
            return f"{d}: bad audio"
        return None

    with ThreadPoolExecutor(workers) as pool:
        errors = [e for e in pool.map(check, doc_ids) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} bad files, e.g. {errors[:5]}")


def _write(ds: Dataset, d: Path, num_shards: int = 1) -> None:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i in range(num_shards):
        shard = ds.shard(num_shards, i, contiguous=True)
        embedded = shard.with_format("arrow").map(embed_table_storage, batched=True)
        embedded.with_format(None).to_parquet(
            d / f"{SPLIT}-{i:05d}-of-{num_shards:05d}.parquet"
        )


def package(
    queries: list[dict],
    judgments: list[dict],
    doc_ids: list[str],
    files: Path,
    out: Path,
) -> None:
    doc_meta = {
        j["doc_id"]: (j["video_language"], j["original_doc_id"]) for j in judgments
    }
    corpus = (
        Dataset.from_dict(
            {
                "id": doc_ids,
                "video": [str(files / f"{d}.mp4") for d in doc_ids],
                "audio": [str(files / f"{d}.m4a") for d in doc_ids],
                "video_language": [doc_meta[d][0] for d in doc_ids],
                "youtube_id": [doc_meta[d][1] for d in doc_ids],
                "has_audio": [not (files / f"{d}{NO_AUDIO}").exists() for d in doc_ids],
            }
        )
        .cast_column("video", Video())
        .cast_column("audio", Audio())
    )
    total = sum(
        (files / f"{d}{e}").stat().st_size for d in doc_ids for e in (".mp4", ".m4a")
    )
    qrels = Dataset.from_dict(
        {
            "query-id": [j["query_id"] for j in judgments if j["relevance"] > 0],
            "corpus-id": [j["doc_id"] for j in judgments if j["relevance"] > 0],
            "score": [j["relevance"] for j in judgments if j["relevance"] > 0],
        }
    ).cast_column("score", Value("int32"))
    _write(Dataset.from_list(queries), out / "queries")
    _write(qrels, out / "data")
    _write(corpus, out / "corpus", num_shards=max(1, -(-total // CORPUS_SHARD_BYTES)))
    card = CARD.format(
        src_url=f"https://huggingface.co/datasets/{SRC_REPO}",
        src_rev=SRC_REVISION,
        n_queries=len(queries),
        n_docs=len(doc_ids),
    )
    (out / "README.md").write_text(card)
    print(
        f"wrote {out}: {len(queries)} queries, {len(doc_ids)} videos, {len(qrels)} qrels"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/multivent2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--repo-id", default="mteb/MultiVENT2")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    queries, judgments = read_metadata(work)
    query_ids = {q["id"] for q in queries}
    assert all(j["query_id"] in query_ids for j in judgments)
    doc_ids = sorted({j["doc_id"] for j in judgments}, key=int)
    print(f"{len(queries)} queries, {len(judgments)} judgments, {len(doc_ids)} videos")

    files = work / "files"
    download(doc_ids, files, args.workers)
    fill_silent_audio(doc_ids, files, args.workers)
    strip_video_audio(doc_ids, files, args.workers)
    verify(doc_ids, files, args.workers)
    out = work / "hf"
    package(queries, judgments, doc_ids, files, out)

    if args.push:
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_large_folder(
            folder_path=str(out), repo_id=args.repo_id, repo_type="dataset"
        )
        print(f"{args.repo_id} revision: {api.dataset_info(args.repo_id).sha}")


if __name__ == "__main__":
    main()
