#!/usr/bin/env python3
"""Build the bilingual (English/Chinese) VATEX retrieval tasks for MTEB.

mteb already has English VATEX tasks, built from the *test* split. VATEX publishes
Chinese captions only for the *validation* split, and the two splits are disjoint - the
5,162 hosted test clips share no ids with the 3,000 validation clips - so the Chinese
side cannot be added to the existing tasks and needs its own build.

Two properties of the source matter:

1. The `url` column of `lmms-lab/vatex_from_url` is malformed. It appends the whole
   clip id to a watch URL, e.g. `...?v=G9zN5TTuGO4_000179_000189`, where only
   `G9zN5TTuGO4` is the video id and `000179_000189` is the second range. The id is
   parsed rather than the URL used.

2. The published archive contains **untrimmed source videos**, not the annotated
   10-second windows. Sampled clips run 15s to 406s against a 10s annotation, so a
   caption would describe a few percent of the retrieved video. Every clip is cut to
   its annotated range before publishing.

Note on frame rate: the encoder must be given the source rate as an exact Fraction.
Rounding 30000/1001 to an int puts the encoder in a different time base from the
incoming frames and every mux fails with EINVAL.

Examples:
  # Extract the annotated clips from the remote archive (ranged reads, no full download).
  uv run python scripts/data/vatex_multilingual_retrieval/create_data.py --stage extract

  # Cut each clip to its annotated window.
  uv run python scripts/data/vatex_multilingual_retrieval/create_data.py --stage trim

  # Publish the shared video corpus plus the en/zh caption configs.
  uv run python scripts/data/vatex_multilingual_retrieval/create_data.py --stage push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import av
from datasets import Dataset, Video
from huggingface_hub import HfApi
from remotezip import RemoteZip

_ANNOTATIONS = "lmms-lab/vatex_from_url"
_ANNOTATION_CONFIG = "vatex_val_zh"
_ARCHIVE_URL = (
    "https://huggingface.co/datasets/zrchen03/vatex_videos_2581/"
    "resolve/main/VATEX_EVAL_VIDEOS_2581.zip"
)
_TARGET_REPO = "vnahata/vatex-multilingual-retrieval"
_PAPER = "https://arxiv.org/abs/1904.03493"
_SPLIT = "test"
_TARGET_CLIPS = 1000  # matches the 1k cap of the existing English VATEX tasks

_CLIP_ID = re.compile(r"^(.+)_(\d{6})_(\d{6})$")


def stage_extract(work: Path) -> dict[str, dict]:
    """Pull the clips that carry both languages out of the remote archive."""
    from datasets import load_dataset

    videos = work / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    ann = load_dataset(_ANNOTATIONS, _ANNOTATION_CONFIG, split="validation")
    caps = {
        r["videoID"]: {"enCap": r["enCap"], "chCap": r["chCap"]}
        for r in ann
        if r.get("enCap") and r.get("chCap")
    }

    with RemoteZip(_ARCHIVE_URL) as z:
        members = {
            os.path.splitext(os.path.basename(n))[0]: n
            for n in z.namelist()
            if n.lower().endswith(".mp4")
        }
        chosen = sorted(set(members) & set(caps))[:_TARGET_CLIPS]
        (work / "annotations.json").write_text(
            json.dumps({v: caps[v] for v in chosen}, ensure_ascii=False),
            encoding="utf-8",
        )
        for i, vid in enumerate(chosen, 1):
            dest = videos / f"{vid}.mp4"
            if dest.exists() and dest.stat().st_size:
                continue
            with z.open(members[vid]) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            if i % 100 == 0:
                print(f"  extracted {i}/{len(chosen)}", flush=True)
    return caps


def _trim_one(src: Path, dst: Path, start: float, end: float) -> bool:
    with av.open(str(src)) as inp:
        if not inp.streams.video:
            return False
        vs = inp.streams.video[0]
        rate = vs.average_rate or Fraction(25, 1)  # exact; see module docstring
        with av.open(str(dst), "w") as out:
            ov = out.add_stream("libx264", rate=rate)
            ov.width = vs.codec_context.width
            ov.height = vs.codec_context.height
            ov.pix_fmt = "yuv420p"
            if vs.time_base:
                inp.seek(int(start / vs.time_base), stream=vs)
            wrote = 0
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t < start:
                    continue
                if t > end:
                    break
                out.mux(ov.encode(frame))
                wrote += 1
            out.mux(ov.encode(None))
    return wrote > 0 and dst.exists() and dst.stat().st_size > 0


def stage_trim(work: Path) -> None:
    """Cut each clip to the window its captions describe."""
    src_dir, dst_dir = work / "videos", work / "trimmed"
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    t0 = time.time()
    sources = sorted(src_dir.glob("*.mp4"))
    for i, src in enumerate(sources, 1):
        m = _CLIP_ID.match(src.stem)
        dst = dst_dir / src.name
        if not m:
            failed += 1
            continue
        if dst.exists() and dst.stat().st_size:
            ok += 1
            continue
        try:
            ok += 1 if _trim_one(src, dst, float(m.group(2)), float(m.group(3))) else 0
        except Exception as e:  # noqa: BLE001
            failed += 1
            if dst.exists():
                dst.unlink()
            print(f"  skip {src.name}: {type(e).__name__} {str(e)[:70]}", flush=True)
        if i % 100 == 0:
            print(f"  trimmed {i}/{len(sources)} ok={ok} failed={failed}", flush=True)
    print(f"trim done: ok={ok} failed={failed} in {time.time() - t0:.0f}s")


def stage_push(work: Path) -> None:
    """Publish one shared video corpus plus a caption config per language."""
    ann = json.loads((work / "annotations.json").read_text(encoding="utf-8"))
    trimmed = work / "trimmed"
    present = sorted(v for v in ann if (trimmed / f"{v}.mp4").exists())
    print(f"publishing {len(present)} clips (of {len(ann)} annotated)")

    api = HfApi()
    api.create_repo(_TARGET_REPO, repo_type="dataset", exist_ok=True)

    videos = Dataset.from_dict(
        {"id": present, "video": [str(trimmed / f"{v}.mp4") for v in present]}
    ).cast_column("video", Video())
    videos.push_to_hub(
        _TARGET_REPO, config_name="videos", split=_SPLIT, max_shard_size="500MB"
    )

    for cfg, key in (("en", "enCap"), ("zh", "chCap")):
        ids, texts, vids = [], [], []
        for v in present:
            caps = ann[v].get(key) or []
            if caps and caps[0].strip():
                ids.append(f"{v}-{cfg}")
                texts.append(caps[0].strip())  # first caption, as the English tasks do
                vids.append(v)
        Dataset.from_dict({"id": ids, "text": texts, "video_id": vids}).push_to_hub(
            _TARGET_REPO, config_name=cfg, split=_SPLIT
        )
        print(f"  {cfg}: {len(ids)} captions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["extract", "trim", "push", "all"], default="all"
    )
    parser.add_argument("--work-dir", type=Path, default=Path("vatex_work"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_PAPER} | archive: {_ARCHIVE_URL}")

    if args.stage in ("extract", "all"):
        stage_extract(args.work_dir)
    if args.stage in ("trim", "all"):
        stage_trim(args.work_dir)
    if args.stage in ("push", "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
