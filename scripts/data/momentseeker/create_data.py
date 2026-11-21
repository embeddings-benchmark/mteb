"""Build the MomentSeeker dataset for MTEB (chunked paper protocol and full videos).

MomentSeeker (Yuan et al., NeurIPS 2025, arXiv:2502.12558) is a long-video moment
retrieval benchmark. Its retrieval protocol (paper Sec. 4.1 and App. 6.1.1) is:

  * divide each long video into fixed 10-second chunks;
  * for a query, rank the chunks *of that query's own video*;
  * a chunk is correct if its temporal IoU with an annotated answer moment is >= 0.3;
  * score with mAP@5 (main metric) and R@1, reported per moment level
    (global / event / object, Sec. 3.3) and overall.

This script reproduces that protocol as an MTEB reranking dataset, and stores the
full-video variant in the same repo so both share one copy of the queries:

  corpus                      10 s chunks of the 271 source videos, ``_id`` =
                              ``<video>_<chunk index:03d>``, chunk k covers
                              [10k, min(10k + 10, duration)).
  <dir>-<level>-queries       the queries, copied unchanged from
                              ``dukesun99/MomentSeeker-Full`` (text, plus the
                              reference image for ti2v or reference clip for tv2v).
  <dir>-<level>-top_ranked    per query, every chunk of its own video, in time order.
  <dir>-<level>-qrels         chunks with IoU >= 0.3 against any answer interval, score 1.
  full-corpus                 the 271 full videos, copied unchanged from
                              ``dukesun99/MomentSeeker-Full`` (``corpus``).
  <dir>-<level>-full-qrels    video-level qrels, copied unchanged from
                              ``dukesun99/MomentSeeker-Full`` (``<dir>-<level>-qrels``).

with <dir> in {t2v, ti2v, tv2v} and <level> in {global, event, object}. The
``MomentSeeker*Reranking`` tasks read the chunk configs and the
``MomentSeeker*FullVideoRetrieval`` tasks read ``full-corpus`` and ``*-full-qrels``.

Queries with no answerable chunk
--------------------------------
Answer moments are short (median 8 s), so for some queries no 10 s chunk reaches
IoU 0.3: 315/1000 t2v and 65/400 each for ti2v and tv2v. The paper still counts these
queries, and they can only score 0. To keep them, each gets a single qrel row with
score 0 (its best-overlapping chunk). MTEB keeps queries whose qrels are all zero and
scores them as misses, so per-level averages run over the same query set as the paper.

Sources
-------
  annotations  avery00/MomentSeeker  {t2v,ti2v,tv2v}.json  (CC BY-NC-SA 4.0)
  videos       dukesun99/MomentSeeker-Full  corpus  (the avery00 source videos,
               re-encoded to 360p with audio)
  queries      dukesun99/MomentSeeker-Full  <dir>-<level>-queries. These follow the
               annotation JSON order filtered by level, with ids q-<position in the whole <dir>.json>;
               the script checks that every query's text and video match the JSON.

Usage:
  python scripts/data/momentseeker/create_data.py --work work/ms
  python scripts/data/momentseeker/create_data.py --work work/ms --push
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Sequence, Value, Video, load_dataset
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
from tqdm import tqdm

ANN_REPO = "avery00/MomentSeeker"
ANN_REVISION = "4a842253a60954cd8765b3d25f6274e2af56b2ff"
FULL_REPO = "dukesun99/MomentSeeker-Full"
FULL_REVISION = "e1be36f6a27a0da618704860cb50517634c01f3a"
SPLIT = "test"

CHUNK_S = 10.0
IOU_THRESHOLD = 0.3
MIN_LAST_CHUNK_S = 1.0  # drop a trailing remainder shorter than this
CORPUS_SHARDS = 10  # ~1 GB parquet files
DIRECTIONS = ("t2v", "ti2v", "tv2v")
LEVELS = ("global", "event", "object")
LEVEL_OF_TASK = {  # paper Sec. 3.3
    "Causal Reasoning": "global",
    "Spatial Relation": "global",
    "Description Location": "event",
    "Action Recognition": "event",
    "Anomaly Detection": "event",
    "Object Recognition": "object",
    "Object Location": "object",
    "Attribute Recognition": "object",
    "OCR": "object",
}


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def video_id(src_video_path: str) -> str:
    return Path(src_video_path).stem  # "./videos/movie101_77.mp4" -> "movie101_77"


# ------------------------------------------------------------ stage 1: inputs
def fetch_inputs(work: Path) -> tuple[dict[str, list[dict]], Path]:
    ann = {}
    for d in DIRECTIONS:
        p = hf_hub_download(
            ANN_REPO, f"{d}.json", repo_type="dataset", revision=ANN_REVISION
        )
        ann[d] = json.loads(Path(p).read_text())
    full = Path(
        snapshot_download(
            FULL_REPO,
            repo_type="dataset",
            revision=FULL_REVISION,
            local_dir=work / "full",
            max_workers=4,
        )
    )
    return ann, full


def split_by_level(ann: dict[str, list[dict]], full: Path) -> dict[str, list[dict]]:
    """Group annotations into the 9 subsets and attach the MomentSeeker-Full query ids."""
    subsets = {}
    for d in DIRECTIONS:
        for lv in LEVELS:
            # MomentSeeker-Full ids are positions in the whole <dir>.json, not within the level
            rows = [
                (i, x) for i, x in enumerate(ann[d]) if LEVEL_OF_TASK[x["task"]] == lv
            ]
            name = f"{d}-{lv}"
            q = pq.read_table(
                next((full / f"{name}-queries").glob("*.parquet")),
                columns=["id", "text"],
            ).to_pylist()
            r = pq.read_table(
                next((full / f"{name}-qrels").glob("*.parquet"))
            ).to_pylist()
            assert len(q) == len(rows) == len(r), name
            gold = {x["query-id"]: x["corpus-id"] for x in r}
            for (i, x), qq in zip(rows, q):
                assert qq["id"] == f"q-{i:04d}", (name, i, qq["id"])
                assert qq["text"] == x["qry_text"], (name, i)
                assert gold[qq["id"]] == video_id(x["src_video_path"]), (name, i)
            subsets[name] = [
                {
                    "id": f"q-{i:04d}",
                    "video": video_id(x["src_video_path"]),
                    "intervals": [
                        tuple(map(float, iv)) for iv in x["answering_time_interval"]
                    ],
                }
                for i, x in rows
            ]
    return subsets


# ------------------------------------------------------------ stage 2: chunks
def duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(p)],
        check=True, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    return float(r.stdout.strip())


def chunk_video(vid: str, src: Path, out_dir: Path) -> list[dict]:
    """Cut one video into 10 s chunks, one accurate-seek ffmpeg call per chunk.

    The segment muxer is not used: on videos whose source keyframes are not on the
    10 s grid it misses the first boundary and shifts every later chunk.
    """
    done = out_dir / vid / "chunks.json"
    if done.exists():
        return json.loads(done.read_text())
    tmp = out_dir / vid / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    dur = duration(src)
    n = int(dur // CHUNK_S) + (dur % CHUNK_S >= MIN_LAST_CHUNK_S)
    chunks = []
    for k in range(n):
        start = k * CHUNK_S
        end = min(start + CHUNK_S, dur)
        part = tmp / f"{k:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{start:g}", "-i", str(src), "-t", f"{end - start:.3f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart", str(part),
            ],
            check=True, capture_output=True, timeout=600,
        )  # fmt: skip
        real = duration(part)
        assert abs(real - (end - start)) < 0.6, (vid, k, real, start, end)
        dst = out_dir / vid / f"{vid}_{k:03d}.mp4"
        part.rename(dst)
        chunks.append({"_id": dst.stem, "path": str(dst), "start": start, "end": end})
    shutil.rmtree(tmp)
    done.write_text(json.dumps(chunks))
    return chunks


def chunk_all(full: Path, work: Path, workers: int) -> dict[str, list[dict]]:
    vids_dir = work / "videos"
    vids_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_dataset(
        "parquet",
        data_files=sorted(str(p) for p in (full / "corpus").glob("*.parquet")),
        split="train",
    ).cast_column("video", Video(decode=False))
    for row in tqdm(corpus, desc="extract source videos"):
        dst = vids_dir / f"{row['_id'] if '_id' in row else row['id']}.mp4"
        if not dst.exists():
            dst.write_bytes(row["video"]["bytes"])
    out = work / "chunks"
    result: dict[str, list[dict]] = {}
    srcs = sorted(vids_dir.glob("*.mp4"))
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = {ex.submit(chunk_video, p.stem, p, out): p.stem for p in srcs}
    try:
        for f in tqdm(as_completed(futs), total=len(futs), desc="chunk"):
            result[futs[f]] = f.result()
    except BaseException:
        # fail fast: without this the pool waits for every queued video first
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return result


# ----------------------------------------------------------- stage 3: labels
def label(subsets: dict[str, list[dict]], chunks: dict[str, list[dict]]):
    tables = {}
    for name, qs in subsets.items():
        qrels, top, unanswerable = [], [], 0
        for q in qs:
            cs = chunks[q["video"]]
            best = {
                c["_id"]: max(iou((c["start"], c["end"]), iv) for iv in q["intervals"])
                for c in cs
            }
            rel = [cid for cid, v in best.items() if v >= IOU_THRESHOLD]
            if rel:
                qrels += [(q["id"], cid, 1) for cid in rel]
            else:
                unanswerable += 1
                qrels.append((q["id"], max(best, key=best.get), 0))
            top.append((q["id"], [c["_id"] for c in cs]))
        rel_per_q = Counter(q for q, _, s in qrels if s > 0)
        print(
            f"{name:12s} queries={len(qs):4d} videos={len({q['video'] for q in qs}):3d} "
            f"unanswerable={unanswerable:3d} "
            f"relevant/answerable={sum(rel_per_q.values()) / max(len(rel_per_q), 1):.2f} "
            f"candidates/query={sum(len(c) for _, c in top) / len(top):.0f}"
        )
        tables[name] = (qrels, top)
    return tables


# --------------------------------------------------------- stage 4: package
CARD = """---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: MomentSeeker
---

# MomentSeeker

Long-video moment retrieval from [MomentSeeker](https://arxiv.org/abs/2502.12558)
(NeurIPS 2025), packaged for [MTEB](https://github.com/embeddings-benchmark/mteb) in
two versions that share the same queries.

Subsets: `{{t2v,ti2v,tv2v}}-{{global,event,object}}`, each with a `-queries` config.

**Chunked (paper protocol), `MomentSeeker*Reranking`.** Each long video is cut into
fixed 10 s chunks (`corpus`, {n_chunks} chunks from {n_videos} videos), and a query ranks
the chunks of its own video (`<subset>-top_ranked`). A chunk is relevant if its temporal
IoU with an annotated answer moment is at least 0.3 (`<subset>-qrels`). Queries with no
such chunk keep a single score-0 qrel, so they count as misses, as in the paper.

**Full video, `MomentSeeker*FullVideoRetrieval`.** The corpus is the {n_videos} complete
videos (`full-corpus`), and the relevant document is the video that contains the answer
moment (`<subset>-full-qrels`).

Built by `scripts/data/momentseeker/create_data.py` in MTEB from
`avery00/MomentSeeker` annotations (revision `{ann_rev}`) and the 360p videos and
query media of `dukesun99/MomentSeeker-Full` (revision `{full_rev}`).

## License

CC BY-NC-SA 4.0, following the source dataset. This dataset is for research only and
not for commercial use. The source authors do not own the copyright of the raw videos.
Copyright holders may ask for removal: contact the MomentSeeker authors
(hyyuan@ruc.edu.cn), and removed videos will also be removed here.

## Citation

```bibtex
@inproceedings{{yuan2025momentseeker,
  author = {{Yuan, Huaying and Ni, Jian and Liu, Zheng and Wang, Yueze and Zhou, Junjie and Liang, Zhengyang and Zhao, Bo and Cao, Zhao and Wen, Ji-Rong and Dou, Zhicheng}},
  booktitle = {{Advances in Neural Information Processing Systems 38}},
  title = {{MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval}},
  year = {{2025}},
}}
```
"""


def _write(
    ds: Dataset, out: Path, cfg: str, embed: bool = False, shards: int = 1
) -> None:
    d = out / cfg
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i in range(shards):
        part = ds.shard(num_shards=shards, index=i, contiguous=True)
        if embed:
            part = part.with_format("arrow").map(embed_table_storage, batched=True)
            part = part.with_format(None)
        part.to_parquet(d / f"{SPLIT}-{i:05d}-of-{shards:05d}.parquet")


def package(full, chunks, tables, out: Path):
    all_chunks = [c for vid in sorted(chunks) for c in chunks[vid]]
    corpus = (
        Dataset.from_dict(
            {
                "_id": [c["_id"] for c in all_chunks],
                "video": [c["path"] for c in all_chunks],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("video", Video())
    )
    _write(corpus, out, "corpus", embed=True, shards=CORPUS_SHARDS)
    configs = ["corpus", "full-corpus"]
    full_corpus = out / "full-corpus"
    shutil.rmtree(full_corpus, ignore_errors=True)
    shutil.copytree(full / "corpus", full_corpus)
    for name, (qrels, top) in tables.items():
        src = next((full / f"{name}-queries").glob("*.parquet"))
        (out / f"{name}-queries").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            src, out / f"{name}-queries" / f"{SPLIT}-00000-of-00001.parquet"
        )
        _write(
            Dataset.from_dict(
                {
                    "query-id": [q for q, _, _ in qrels],
                    "corpus-id": [c for _, c, _ in qrels],
                    "score": [s for _, _, s in qrels],
                }
            ).cast_column("score", Value("int32")),
            out,
            f"{name}-qrels",
        )
        _write(
            Dataset.from_dict(
                {"query-id": [q for q, _ in top], "corpus-ids": [c for _, c in top]}
            ).cast_column("corpus-ids", Sequence(Value("string"))),
            out,
            f"{name}-top_ranked",
        )
        shutil.rmtree(out / f"{name}-full-qrels", ignore_errors=True)
        shutil.copytree(full / f"{name}-qrels", out / f"{name}-full-qrels")
        configs += [
            f"{name}-queries",
            f"{name}-qrels",
            f"{name}-top_ranked",
            f"{name}-full-qrels",
        ]
    yaml = "configs:\n" + "".join(
        f"- config_name: {c}\n  data_files:\n  - split: {SPLIT}\n    path: {c}/{SPLIT}-*\n"
        for c in configs
    )
    card = CARD.format(
        n_chunks=len(all_chunks),
        n_videos=len(chunks),
        ann_rev=ANN_REVISION,
        full_rev=FULL_REVISION,
    )
    card = card.replace("---\n\n#", yaml + "---\n\n#", 1)
    (out / "README.md").write_text(card)
    print(f"wrote {out}: {len(all_chunks)} chunks from {len(chunks)} videos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/momentseeker")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repo-id", default="mteb/MomentSeeker")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ann, full = fetch_inputs(work)
    subsets = split_by_level(ann, full)
    chunks = chunk_all(full, work, args.workers)
    missing = {q["video"] for qs in subsets.values() for q in qs} - set(chunks)
    assert not missing, f"query videos missing from corpus: {sorted(missing)[:5]}"
    for qs in subsets.values():  # every answer interval must lie inside its video
        for q in qs:
            end = chunks[q["video"]][-1]["end"]
            assert all(b <= end + CHUNK_S for _, b in q["intervals"]), (q, end)
    tables = label(subsets, chunks)
    out = work / "hf"
    package(full, chunks, tables, out)

    if args.push:
        create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_large_folder(
            folder_path=str(out), repo_id=args.repo_id, repo_type="dataset"
        )
        print(f"{args.repo_id} revision: {api.dataset_info(args.repo_id).sha}")


if __name__ == "__main__":
    main()
