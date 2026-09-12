#!/usr/bin/env python3
"""Build the Jaco Play image<->video retrieval tasks for MTEB.

Follows the construction already merged for LIBERO and ManiSkill: the query is the goal
state of a held-out episode (its final frame) and the corpus is other episodes' videos,
with a corpus item relevant iff it comes from the same task. Held-out episodes never
appear in the corpus, so the task cannot be solved by matching identical frames, and
episodes recorded in the same scene that accomplish a different goal stay non-relevant so
matching the scene alone is not enough.

Tasks with fewer than `--min-episodes` demonstrations are dropped, so every query has
several distinct relevant videos rather than a single one.

Two source details matter:

1. LeRobot concatenates all episodes into one mp4 per camera and addresses each episode
   by a `[from_timestamp, to_timestamp]` range, so clips are cut by timestamp rather than
   read as separate files.

2. The encoder must be given the source frame rate as an exact `Fraction`. Rounding it to
   an int puts the encoder in a different time base from the incoming frames and every
   mux fails with EINVAL.

The v2i direction is the same media with the roles swapped: videos become queries, goal
images become the corpus, and the qrels are reversed.

Examples:
  # Build both directions locally.
  uv run python scripts/data/jaco_play_retrieval/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/jaco_play_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import collections
import io
import json
from fractions import Fraction
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Image, Video
from huggingface_hub import HfApi, snapshot_download

_SOURCE_REPO = "lerobot/jaco_play"
_HOMEPAGE = "https://github.com/clvrai/clvr_jaco_play_dataset"
_LICENSE = "cc-by-4.0"
_TARGET_I2V = "vnahata/JacoPlay-I2V"
_TARGET_V2I = "vnahata/JacoPlay-V2I"
_SPLIT = "test"

# the scene camera; the wrist camera is too close-up to identify a task from one frame
_VIDEO_KEY = "observation.images.image"


def _episodes(src: Path) -> list[dict]:
    cols = [
        "episode_index",
        "tasks",
        f"videos/{_VIDEO_KEY}/from_timestamp",
        f"videos/{_VIDEO_KEY}/to_timestamp",
    ]
    rows = pq.read_table(
        src / "meta/episodes/chunk-000/file-000.parquet", columns=cols
    ).to_pylist()
    return [
        {
            "episode": r["episode_index"],
            "task": r["tasks"][0],
            "start": r[f"videos/{_VIDEO_KEY}/from_timestamp"],
            "end": r[f"videos/{_VIDEO_KEY}/to_timestamp"],
        }
        for r in rows
        if r["tasks"]
    ]


def _cut(video: Path, start: float, end: float) -> tuple[bytes, bytes] | None:
    """Return (clip mp4 bytes, final frame jpeg bytes) for one episode."""
    with av.open(str(video)) as inp:
        vs = inp.streams.video[0]
        rate = vs.average_rate or Fraction(10, 1)  # exact; see module docstring
        buf, last, wrote = io.BytesIO(), None, 0
        with av.open(buf, "w", format="mp4") as out:
            ov = out.add_stream("libx264", rate=rate)
            ov.width = vs.codec_context.width
            ov.height = vs.codec_context.height
            ov.pix_fmt = "yuv420p"
            if vs.time_base:
                inp.seek(int(start / vs.time_base), stream=vs)
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t < start:
                    continue
                if t >= end:
                    break
                out.mux(ov.encode(frame))
                last, wrote = frame, wrote + 1
            out.mux(ov.encode(None))
        if not wrote or last is None:
            return None
    img = io.BytesIO()
    last.to_image().save(img, format="JPEG", quality=92)
    return buf.getvalue(), img.getvalue()


def stage_build(work: Path, min_episodes: int, per_task: int) -> dict:
    src = Path(snapshot_download(_SOURCE_REPO, repo_type="dataset"))
    out_i2v, out_v2i = work / "i2v", work / "v2i"
    out_i2v.mkdir(parents=True, exist_ok=True)
    out_v2i.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for e in _episodes(src):
        by_task[e["task"]].append(e)
    kept = {
        t: sorted(v, key=lambda x: x["episode"])
        for t, v in by_task.items()
        if len(v) >= min_episodes
    }
    print(f"tasks kept: {len(kept)} of {len(by_task)}")

    video = src / f"videos/{_VIDEO_KEY}/chunk-000/file-000.mp4"
    queries, corpus, qrels, dropped = [], [], [], 0

    for _task, episodes in sorted(kept.items()):
        q_eps, c_eps = episodes[:per_task], episodes[per_task:]
        for e in c_eps:
            cut = _cut(video, e["start"], e["end"])
            if cut is None:
                dropped += 1
                continue
            cid = f"ep{e['episode']:05d}"
            corpus.append({"id": cid, "video": {"bytes": cut[0], "path": f"{cid}.mp4"}})
        for e in q_eps:
            cut = _cut(video, e["start"], e["end"])
            if cut is None:
                dropped += 1
                continue
            qid = f"goal{e['episode']:05d}"
            queries.append(
                {"id": qid, "image": {"bytes": cut[1], "path": f"{qid}.jpg"}}
            )
            qrels.extend(
                {"query-id": qid, "corpus-id": f"ep{c['episode']:05d}", "score": 1}
                for c in c_eps
            )

    pq.write_table(pa.Table.from_pylist(queries), out_i2v / "queries.parquet")
    pq.write_table(pa.Table.from_pylist(corpus), out_i2v / "corpus.parquet")
    pq.write_table(pa.Table.from_pylist(qrels), out_i2v / "qrels.parquet")

    # v2i is the same media with the roles swapped
    pq.write_table(pa.Table.from_pylist(corpus), out_v2i / "queries.parquet")
    pq.write_table(pa.Table.from_pylist(queries), out_v2i / "corpus.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"query-id": r["corpus-id"], "corpus-id": r["query-id"], "score": 1}
                for r in qrels
            ]
        ),
        out_v2i / "qrels.parquet",
    )

    stats = {
        "tasks": len(kept),
        "i2v_queries": len(queries),
        "i2v_corpus": len(corpus),
        "qrels": len(qrels),
        "dropped": dropped,
    }
    (work / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("built", json.dumps(stats))
    return stats


def stage_push(work: Path) -> None:
    api = HfApi()
    for sub, repo, qcol, ccol in (
        ("i2v", _TARGET_I2V, Image(), Video()),
        ("v2i", _TARGET_V2I, Video(), Image()),
    ):
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        d = work / sub
        qname = "image" if isinstance(qcol, Image) else "video"
        cname = "image" if isinstance(ccol, Image) else "video"
        Dataset.from_parquet(str(d / "queries.parquet")).cast_column(
            qname, qcol
        ).push_to_hub(repo, config_name="queries", split=_SPLIT, max_shard_size="400MB")
        Dataset.from_parquet(str(d / "corpus.parquet")).cast_column(
            cname, ccol
        ).push_to_hub(repo, config_name="corpus", split=_SPLIT, max_shard_size="400MB")
        Dataset.from_parquet(str(d / "qrels.parquet")).push_to_hub(
            repo, config_name="qrels", split=_SPLIT
        )
        print(f"pushed {repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("jaco_work"))
    parser.add_argument("--min-episodes", type=int, default=4)
    parser.add_argument("--queries-per-task", type=int, default=2)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_HOMEPAGE} (license {_LICENSE})")

    if args.stage in ("build", "all"):
        stage_build(args.work_dir, args.min_episodes, args.queries_per_task)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
