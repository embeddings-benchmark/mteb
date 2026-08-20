"""Package UCF101 into MTEB i2v retrieval format.

Source: https://huggingface.co/datasets/mteb/UCF101-51VA (video + audio + label,
single "default" config, no ids). Only the *test* split is used, so the official
split-1 boundary is preserved.

Construction
------------
UCF101 clips are grouped by scene: ``v_<Class>_g<NN>_c<NN>``, where ``gNN`` is the
group (same actor / scene / recording session) and ``cNN`` is the clip within it.
Clips inside a group are near-duplicates, so putting several of them in the corpus
would make instance-level qrels unanswerable. Instead:

  corpus  = clip ``c01`` of each group -- exactly ONE clip per group, so the corpus
            contains no intra-group near-duplicates at all.
  queries = one frame from the HIGHEST-index clip of the same group, which is always
            >= 2 clip indices away from c01, sampled uniformly at random from that
            clip's interior (25%-75% of its duration), seeded.
  qrels   = the query for group G is relevant to the c01 of group G only, score 1.

Leakage filter
--------------
Even at a >= 2 clip-index gap, some query frames are near-identical to a frame of
their own positive clip (static-camera classes such as WallPushups). Groups whose
query frame has raw min RMS < ``--leak-threshold`` against ANY frame of its positive
clip are dropped. The default 7.0 is the largest threshold that still leaves every
one of the 51 classes with >= 3 groups.

The clip filename is only available as the hidden ``path`` subfield of the source's
Video feature, so this script reads it via parquet column projection -- which also
avoids downloading the 1.52 GB audio column.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/ucf101_i2v_retrieval/create_data.py \\
      --repo-id hubxrt/UCF101-I2V --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Image, Value, Video
from huggingface_hub import HfApi, HfFileSystem, create_repo
from PIL import Image as PILImage
from tqdm import tqdm

SOURCE_REPO = "mteb/UCF101-51VA"
SOURCE_REVISION = "866b006d84629d66d9927646db89bd43381925e7"
SPLIT = "test"
SEED = 20260819
MIN_CLIPS = 4
MIN_CLIP_INDEX_GAP = 2
INTERIOR_LO, INTERIOR_HI = 0.25, 0.75
CLIP_RE = re.compile(r"^v_([A-Za-z]+)_g(\d+)_c(\d+)\.avi$")


def _index_clips(fs: HfFileSystem, files: list[str]):
    """Map clip filename -> (parquet file, row group); group clips by (class, group)."""
    loc: dict[str, tuple[str, int]] = {}
    groups: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for f in files:
        with fs.open(f"datasets/{SOURCE_REPO}/{f}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            md = pf.metadata
            rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
            paths = (
                pf.read(columns=["video.path"])
                .column("video")
                .combine_chunks()
                .field("path")
                .to_pylist()
            )
        off = 0
        for rg, n in enumerate(rg_rows):
            for p in paths[off : off + n]:
                m = CLIP_RE.match(p)
                if m is None:
                    raise ValueError(f"unexpected clip filename: {p}")
                loc[p] = (f, rg)
                groups[(m.group(1), int(m.group(2)))].append((int(m.group(3)), p))
            off += n
    for v in groups.values():
        v.sort()
    return loc, groups


def _select_groups(groups) -> list[dict]:
    """Groups with >= MIN_CLIPS clips, c01 present, top clip far enough from c01."""
    out = []
    for (cls, gnum), clips in sorted(groups.items()):
        nums = [n for n, _ in clips]
        if len(clips) < MIN_CLIPS or 1 not in nums:
            continue
        top = max(nums)
        if top - 1 < MIN_CLIP_INDEX_GAP:
            continue
        d = dict(clips)
        out.append(
            {
                "group": f"v_{cls}_g{gnum:02d}",
                "cls": cls,
                "corpus_clip": d[1],
                "query_clip": d[top],
                "query_cnum": top,
                "clip_index_gap": top - 1,
                "n_clips_in_group": len(clips),
            }
        )
    return out


def _fetch_clips(fs: HfFileSystem, loc, needed: set[str], work: Path) -> None:
    """Fetch only the row groups holding the needed clips; write the .avi bytes out."""
    want: dict[str, set[int]] = defaultdict(set)
    for p in needed:
        if not (work / p).exists():
            f, rg = loc[p]
            want[f].add(rg)
    if not want:
        return
    total = sum(len(v) for v in want.values())
    with tqdm(total=total, desc="fetching row groups") as bar:
        for f in sorted(want):
            with fs.open(f"datasets/{SOURCE_REPO}/{f}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                for rg in sorted(want[f]):
                    t = pf.read_row_group(rg, columns=["video.bytes", "video.path"])
                    v = t.column("video").combine_chunks()
                    for p, b in zip(
                        v.field("path").to_pylist(), v.field("bytes").to_pylist()
                    ):
                        if p in needed:
                            (work / p).write_bytes(b)
                    del t, v
                    bar.update(1)


def _frames_small(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack(
            [
                f.reformat(width=32, height=32, format="gray").to_ndarray()
                for f in c.decode(video=0)
            ]
        ).astype(np.float64)


def _rms(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _make_query_frame(clip: Path, group: str, out_png: Path) -> tuple[int, int]:
    """Sample one interior frame, seeded per group. Returns (frame index, n frames)."""
    with av.open(str(clip)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n = len(frames)
    lo = max(0, int(np.ceil(INTERIOR_LO * n)))
    hi = min(n - 1, int(np.floor(INTERIOR_HI * n)))
    seed = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:12], 16)
    idx = random.Random(seed).randint(lo, hi)
    PILImage.fromarray(frames[idx]).save(out_png)
    return idx, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="hubxrt/UCF101-I2V")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./ucf101_i2v_work"),
        help="scratch dir for extracted clips and query frames (reused if populated)",
    )
    ap.add_argument("--leak-threshold", type=float, default=7.0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    work = args.work_dir
    for sub in ("corpus", "query", "query_frames"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    api = HfApi()
    fs = HfFileSystem(revision=SOURCE_REVISION)
    files = sorted(
        s.rfilename
        for s in api.repo_info(
            SOURCE_REPO, repo_type="dataset", revision=SOURCE_REVISION
        ).siblings
        if s.rfilename.endswith(".parquet") and f"/{SPLIT}-" in s.rfilename
    )
    loc, groups = _index_clips(fs, files)
    selected = _select_groups(groups)
    print(f"{len(selected)} eligible groups in the {SPLIT} split")

    _fetch_clips(
        fs,
        loc,
        {s["corpus_clip"] for s in selected},
        work / "corpus",
    )
    _fetch_clips(
        fs,
        loc,
        {s["query_clip"] for s in selected},
        work / "query",
    )

    rows = []
    for s in tqdm(selected, desc="query frames + leakage"):
        png = work / "query_frames" / f"{s['group']}.png"
        cpath = work / "corpus" / s["corpus_clip"]
        if png.exists():
            with av.open(str(work / "query" / s["query_clip"])) as c:
                n_q = c.streams.video[0].frames
            idx = json.loads((work / "query_frames" / "index.json").read_text())[
                s["group"]
            ]
        else:
            idx, n_q = _make_query_frame(
                work / "query" / s["query_clip"], s["group"], png
            )
        small = np.array(
            PILImage.open(png).convert("L").resize((32, 32), PILImage.BILINEAR),
            dtype=np.float64,
        )
        corpus_frames = _frames_small(cpath)
        step = statistics.median(
            [
                _rms(corpus_frames[i], corpus_frames[i + 1])
                for i in range(len(corpus_frames) - 1)
            ]
        )
        min_rms = min(_rms(small, f) for f in corpus_frames)
        rows.append(
            {
                **s,
                "query_frame_idx": idx,
                "query_n_frames": n_q,
                "min_rms": min_rms,
                "corpus_median_step": step,
            }
        )
    (work / "query_frames" / "index.json").write_text(
        json.dumps({r["group"]: r["query_frame_idx"] for r in rows})
    )

    kept = [r for r in rows if r["min_rms"] >= args.leak_threshold]
    per_class = defaultdict(int)
    for r in kept:
        per_class[r["cls"]] += 1
    print(
        f"leakage filter (raw min RMS < {args.leak_threshold}): "
        f"{len(rows) - len(kept)} dropped, {len(kept)} kept; "
        f"{len(per_class)} classes, min {min(per_class.values())} groups/class"
    )

    corpus = (
        Dataset.from_dict(
            {
                "_id": [Path(r["corpus_clip"]).stem for r in kept],
                "video": [str(work / "corpus" / r["corpus_clip"]) for r in kept],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("video", Video())
    )
    queries = (
        Dataset.from_dict(
            {
                "_id": [
                    f"{Path(r['query_clip']).stem}_f{r['query_frame_idx']:04d}"
                    for r in kept
                ],
                "image": [str(work / "query_frames" / f"{r['group']}.png") for r in kept],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("image", Image())
    )
    qrels = (
        Dataset.from_dict(
            {
                "query-id": [
                    f"{Path(r['query_clip']).stem}_f{r['query_frame_idx']:04d}"
                    for r in kept
                ],
                "corpus-id": [Path(r["corpus_clip"]).stem for r in kept],
                "score": [1] * len(kept),
            }
        )
        .cast_column("query-id", Value("string"))
        .cast_column("corpus-id", Value("string"))
        .cast_column("score", Value("int32"))
    )
    print(f"corpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")
    print(f"corpus features : {corpus.features}")
    print(f"queries features: {queries.features}")
    print(f"qrels features  : {qrels.features}")

    if not args.push:
        print("\n--push not set; nothing uploaded.")
        return

    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    for ds, cfg in ((corpus, "corpus"), (queries, "queries"), (qrels, "default")):
        ds.push_to_hub(args.repo_id, config_name=cfg, split=SPLIT)
        print(f"pushed config {cfg!r}")
    sha = api.dataset_info(args.repo_id).sha
    print(f"\nrevision: {sha}")


if __name__ == "__main__":
    main()
