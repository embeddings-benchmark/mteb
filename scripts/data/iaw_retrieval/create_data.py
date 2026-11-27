"""Build the IKEA Assembly in the Wild (IAW) clip <-> step-diagram dataset for MTEB.

IAW (Zhang et al., CVPR 2023, arXiv:2303.13800) pairs YouTube videos of people assembling
IKEA furniture with the furniture's assembly manual. Each manual step diagram is boxed on
its manual page, and each video is annotated with the time segments in which every step
is carried out. The paper evaluates two tasks on the test split, both inside one
furniture item: video-to-diagram retrieval (a 10 s clip ranks the step diagrams of its
manual; top-1 accuracy) and diagram-to-video retrieval (a step diagram ranks the clips of
its video; recall@1/3).

This script rebuilds that test protocol:

  * annotations: ``dataset/split/test.json`` (226 items, one video each) and
    ``test_clip.csv`` (the paper's 11,103 test clips of 10 s) from the dataset repo at a
    pinned commit;
  * step diagrams: every manual page is rendered at 150 dpi with PyMuPDF and the step
    boxes are cropped exactly as the dataset's ``get_dataset.py`` does;
  * clips: each video is downloaded at up to 360p and cut into the paper's clips
    (silent H.264).

The source only ships URLs, and many IKEA manual PDFs have since been taken down. A manual
is fetched from its original URL and, failing that, from its Internet Archive snapshot of
the same URL (the file name carries the manual's revision, so the snapshot is the
annotated file). An item is kept only if all of its manuals and its video can be
fetched, every annotated page exists, and the video's length still matches the
annotation; dropped items are listed with the reason.

Configs (split ``test``), in the video-to-diagram direction:

  queries     the clips: ``id``, ``video``, ``furniture_id``, ``youtube_id``, ``step``,
              ``start_time``, ``end_time``.
  corpus      the step diagrams: ``id``, ``image``, ``furniture_id``, ``step``,
              ``manual_url``, ``manual_source`` (``live`` or ``wayback``), ``page``.
  default     qrels: clip -> the diagram of its step.
  top_ranked  clip -> the step diagrams of its own manual (the candidates).

The diagram-to-video task swaps the roles in ``dataset_transform``.

Requirements beyond ``datasets``: ``pymupdf``, ``yt-dlp`` (with a JavaScript runtime such
as node) and ``ffmpeg``.

Usage:
  python scripts/data/iaw_retrieval/create_data.py --work work/iaw
  python scripts/data/iaw_retrieval/create_data.py --work work/iaw --push
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import requests
from datasets import Dataset, Image, Sequence, Value, Video
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo
from PIL import Image as PILImage
from tqdm import tqdm

SRC_REPO = "DavidZhang73/IKEAAssemblyInTheWildDataset"
SRC_COMMIT = "3ad88a349b93ebf889cc2d759e27634f0a87b3cc"
RAW_URL = f"https://raw.githubusercontent.com/{SRC_REPO}/{SRC_COMMIT}/dataset/split"
REPO_ID = "mteb/IAW"
SPLIT = "test"
DPI = 150  # the resolution the step boxes were drawn at
MAX_HEIGHT = 360
DURATION_TOLERANCE = (
    2.0  # seconds between the annotated and the downloaded video length
)
SHARD_BYTES = 1_000_000_000
pymupdf.TOOLS.mupdf_display_errors(False)
HEADERS = {"User-Agent": "Mozilla/5.0 (dataset build script)"}

CARD = """---
license: mit
language:
- en
tags:
- mteb
- video
- image
- retrieval
size_categories:
- 10K<n<100K
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
- config_name: top_ranked
  data_files:
  - split: test
    path: top_ranked/test-*.parquet
---

# IKEA Assembly in the Wild (IAW), test split for MTEB

The test split of [IKEA Assembly in the Wild](https://github.com/{src_repo}) (commit
`{src_commit_short}`), re-packaged for [MTEB](https://github.com/embeddings-benchmark/mteb)
under the paper's test protocol: video clips of people assembling furniture matched to the
step diagrams of the furniture's assembly manual, within one furniture item.

- **queries:** {n_clips:,} clips of 10 s (the paper's test clips; silent, up to {max_h}p).
- **corpus:** {n_steps:,} step diagrams, cropped from manual pages rendered at {dpi} dpi.
- **qrels (`default`):** each clip -> the step it shows.
- **top_ranked:** each clip -> the step diagrams of its own manual.

{n_items} of the 226 test items are kept: an item is dropped when one of its manuals or
its video is no longer available, or the video no longer matches the annotation. Manuals
missing from IKEA's site were taken from the Internet Archive's copy of the same URL
(`manual_source`).

The annotations are MIT licensed. The manuals are IKEA's and the videos belong to their
YouTube uploaders. The build script is `scripts/data/iaw_retrieval/create_data.py` in the
MTEB repository.

## Citation

```bibtex
@inproceedings{{zhang2023aligning,
  author = {{Zhang, Jiahao and Cherian, Anoop and Liu, Yanbin and Ben-Shabat, Yizhak and Rodriguez, Cristian and Gould, Stephen}},
  booktitle = {{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}},
  title = {{Aligning Step-by-Step Instructional Diagrams to Video Demonstrations}},
  year = {{2023}},
}}
```
"""


def fetch_annotations(work: Path) -> tuple[list[dict], list[dict]]:
    ann = work / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for name in ["test.json", "test_clip.csv"]:
        if not (ann / name).exists():
            r = requests.get(f"{RAW_URL}/{name}", timeout=120)
            r.raise_for_status()
            (ann / name).write_bytes(r.content)
    items = json.loads((ann / "test.json").read_text())
    with open(ann / "test_clip.csv", newline="") as f:
        clips = list(csv.DictReader(f))
    return items, clips


class TransientError(Exception):
    """A request that failed but may succeed later; its result is not cached."""


_ARCHIVE_LOCK = threading.Lock()
_ARCHIVE_GAP = 2.0  # seconds between Internet Archive requests
_last_archive_call = [0.0]


def _request(
    url: str, params: dict | None = None, archive: bool = False
) -> requests.Response:
    """GET with retries.

    Internet Archive requests are spaced out: it refuses connections for a while from
    clients that go too fast.
    """
    waits = [30, 120, 300] if archive else [5, 15, 30]
    for wait in [*waits, None]:
        if archive:
            with _ARCHIVE_LOCK:
                pause = _last_archive_call[0] + _ARCHIVE_GAP - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
                _last_archive_call[0] = time.monotonic()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            if r.status_code not in (429, 500, 502, 503, 504):
                return r
        except requests.RequestException:
            pass
        if wait is None:
            break
        time.sleep(wait)
    raise TransientError(url)


def _get_pdf(url: str, archive: bool = False) -> bytes | None:
    """Return the PDF at ``url``, or None if it is missing or damaged.

    Some Internet Archive captures are truncated; MuPDF then has to repair the file and
    pages come out blank or incomplete, so such files are rejected.
    """
    r = _request(url, archive=archive)
    if r.status_code != 200 or r.content[:5] != b"%PDF-":
        return None
    with pymupdf.open(stream=r.content, filetype="pdf") as doc:
        if doc.is_repaired or doc.page_count == 0:
            return None
    return r.content


def _wayback_snapshots(url: str) -> list[str]:
    """Timestamps of the Internet Archive's 200-status captures of ``url``, newest first."""
    r = _request(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": url,
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp",
        },
        archive=True,
    )
    if r.status_code != 200:
        raise TransientError(url)
    rows = r.json() if r.text.strip() else []
    return sorted((row[0] for row in rows[1:]), reverse=True)


def fetch_manual(url: str, dest: Path) -> dict:
    """Download a manual PDF, falling back to its Internet Archive capture.

    Only definite outcomes are cached, so a rerun retries failed lookups.
    """
    meta = dest.with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text())
    info = {"url": url, "source": None}
    try:
        data = _get_pdf(url)
        if data is not None:
            info["source"] = "live"
        else:
            for ts in _wayback_snapshots(url)[:5]:
                data = _get_pdf(
                    f"https://web.archive.org/web/{ts}id_/{url}", archive=True
                )
                if data is not None:
                    info.update(source="wayback", snapshot=ts)
                    break
    except TransientError as e:
        return {**info, "error": f"request failed: {e}"}
    if data is not None:
        dest.write_bytes(data)
    meta.write_text(json.dumps(info))
    return info


def crop_steps(item: dict, manual_paths: list[Path], out_dir: Path) -> list[Path] | str:
    """Crop every step box of ``item`` the way the dataset's get_dataset.py does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"step-{i:02d}.png" for i in range(len(item["annotationList"]))]
    if all(p.exists() for p in paths):
        return paths
    docs = [pymupdf.open(p) for p in manual_paths]
    try:
        pages: dict[tuple[int, int], PILImage.Image] = {}
        for path, step in zip(paths, item["annotationList"]):
            key = (step["manual"], step["page"])
            if step["page"] >= len(docs[step["manual"]]):
                return f"page {step['page']} missing from manual {step['manual']}"
            if key not in pages:
                pix = docs[step["manual"]][step["page"]].get_pixmap(dpi=DPI)
                pages[key] = PILImage.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples
                )
            page = pages[key]
            box = (
                max(0, step["x"]),
                max(0, step["y"]),
                min(step["x"] + step["width"], page.width),
                min(step["y"] + step["height"], page.height),
            )
            crop = page.crop(box)
            lo, hi = crop.convert("L").getextrema()
            if lo == 255 or hi == 0:
                return f"step {step['step']} diagram is blank"
            crop.save(path)
    finally:
        for doc in docs:
            doc.close()
    return paths


def fetch_video(url: str, dest_dir: Path) -> Path | None:
    vid = url.split("watch?v=")[-1]
    out = dest_dir / f"{vid}.mp4"
    if out.exists():
        return out
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--js-runtimes",
        "node",
        "-f",
        f"bv*[height<={MAX_HEIGHT}][ext=mp4]/bv*[height<={MAX_HEIGHT}]",
        "--remux-video",
        "mp4",
        "-o",
        str(dest_dir / f"{vid}.part.%(ext)s"),
        url,
    ]
    for _ in range(3):
        res = subprocess.run(cmd, capture_output=True, text=True)
        part = dest_dir / f"{vid}.part.mp4"
        if res.returncode == 0 and part.exists():
            part.rename(out)
            return out
        if "unavailable" in res.stderr.lower() or "private" in res.stderr.lower():
            break
        time.sleep(10)
    (dest_dir / f"{vid}.error").write_text(res.stderr[-2000:])
    return None


def probe_duration(path: Path) -> float:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip().split(",")[0])


def cut_clip(src: Path, start: float, end: float, out: Path) -> Path:
    if out.exists():
        return out
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{end - start:.3f}",
            "-an",
            "-vf",
            f"scale=-2:'min({MAX_HEIGHT},ih)'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    tmp.rename(out)
    return out


def step_id(furniture_id: str, step: int) -> str:
    return f"{furniture_id}-step{step:02d}"


def build(work: Path, workers: int, limit: int | None) -> tuple[list, list, dict]:
    items, clip_rows = fetch_annotations(work)
    if limit:
        items = items[:limit]
    dropped: dict[str, str] = {}

    # manuals
    urls = sorted({m["url"] for it in items for m in it["manualList"]})
    man_dir = work / "manuals"
    man_dir.mkdir(parents=True, exist_ok=True)

    def _manual(url: str) -> tuple[str, dict]:
        return url, fetch_manual(url, man_dir / url.split("/")[-1])

    with ThreadPoolExecutor(4) as ex:
        manuals = dict(tqdm(ex.map(_manual, urls), total=len(urls), desc="manuals"))
    by_source = defaultdict(int)
    for info in manuals.values():
        by_source[info["source"]] += 1
    print("manuals:", dict(by_source))

    # step diagrams
    steps: dict[str, list[Path]] = {}
    for it in tqdm(items, desc="steps"):
        missing = [
            m["url"] for m in it["manualList"] if not manuals[m["url"]]["source"]
        ]
        if missing:
            why = manuals[missing[0]].get("error", "manual unavailable")
            dropped[it["id"]] = f"{why}: {missing[0].split('/')[-1]}"
            continue
        paths = [man_dir / m["url"].split("/")[-1] for m in it["manualList"]]
        res = crop_steps(it, paths, work / "steps" / it["id"])
        if isinstance(res, str):
            dropped[it["id"]] = res
            continue
        steps[it["id"]] = res

    # videos
    keep = [it for it in items if it["id"] in steps]
    vid_dir = work / "videos"
    urls = sorted({it["videoList"][0]["url"] for it in keep})
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        videos = dict(
            zip(
                urls,
                tqdm(
                    ex.map(lambda u: fetch_video(u, vid_dir), urls),
                    total=len(urls),
                    desc="videos",
                ),
            )
        )
    kept_items = []
    for it in keep:
        v = it["videoList"][0]
        path = videos[v["url"]]
        if path is None:
            dropped[it["id"]] = "video unavailable"
            continue
        dur = probe_duration(path)
        if abs(dur - v["duration"]) > DURATION_TOLERANCE:
            dropped[it["id"]] = f"video length {dur:.0f} s, annotated {v['duration']} s"
            continue
        kept_items.append(it)

    # clips
    kept_ids = {it["id"] for it in kept_items}
    by_id = {it["id"]: it for it in kept_items}
    rows = [r for r in clip_rows if r["furniture_id"] in kept_ids]
    clip_dir = work / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    def _clip(r: dict) -> dict:
        fid, step, ci = r["furniture_id"], int(r["step_index"]), int(r["clip_index"])
        cid = f"{step_id(fid, step)}-clip{ci:03d}"
        src = videos[by_id[fid]["videoList"][0]["url"]]
        start, end = float(r["start_time"]), float(r["end_time"])
        path = cut_clip(src, start, end, clip_dir / f"{cid}.mp4")
        return {
            "id": cid,
            "video": str(path),
            "furniture_id": fid,
            "youtube_id": r["video_id"],
            "step": step,
            "start_time": start,
            "end_time": end,
        }

    with ThreadPoolExecutor(workers) as ex:
        clips = list(tqdm(ex.map(_clip, rows), total=len(rows), desc="clips"))

    diagrams = []
    for it in kept_items:
        for step, (path, ann) in enumerate(zip(steps[it["id"]], it["annotationList"])):
            m = it["manualList"][ann["manual"]]["url"]
            diagrams.append(
                {
                    "id": step_id(it["id"], step),
                    "image": str(path),
                    "furniture_id": it["id"],
                    "step": step,
                    "manual_url": m,
                    "manual_source": manuals[m]["source"],
                    "page": ann["page"],
                }
            )
    for fid, reason in sorted(dropped.items()):
        print(f"dropped {fid}: {reason}")
    print(
        f"kept {len(kept_items)} of {len(items)} items: {len(clips)} clips, "
        f"{len(diagrams)} diagrams"
    )
    return clips, diagrams, {"n_items": len(kept_items), "dropped": dropped}


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


def package(clips: list[dict], diagrams: list[dict], info: dict, out: Path) -> None:
    queries = Dataset.from_list(clips).cast_column("video", Video())
    corpus = Dataset.from_list(diagrams).cast_column("image", Image())
    qrels = Dataset.from_dict(
        {
            "query-id": [c["id"] for c in clips],
            "corpus-id": [step_id(c["furniture_id"], c["step"]) for c in clips],
            "score": [1] * len(clips),
        }
    ).cast_column("score", Value("int32"))
    manual_steps: dict[str, list[str]] = defaultdict(list)
    for d in diagrams:
        manual_steps[d["furniture_id"]].append(d["id"])
    top_ranked = Dataset.from_dict(
        {
            "query-id": [c["id"] for c in clips],
            "corpus-ids": [manual_steps[c["furniture_id"]] for c in clips],
        }
    ).cast_column("corpus-ids", Sequence(Value("string")))

    _write(
        queries, out / "queries", sum(Path(c["video"]).stat().st_size for c in clips)
    )
    _write(
        corpus, out / "corpus", sum(Path(d["image"]).stat().st_size for d in diagrams)
    )
    _write(qrels, out / "data")
    _write(top_ranked, out / "top_ranked")
    (out / "README.md").write_text(
        CARD.format(
            src_repo=SRC_REPO,
            src_commit_short=SRC_COMMIT[:8],
            n_clips=len(clips),
            n_steps=len(diagrams),
            n_items=info["n_items"],
            max_h=MAX_HEIGHT,
            dpi=DPI,
        )
    )
    (out / "dropped_items.json").write_text(json.dumps(info["dropped"], indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/iaw")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the first N items (for testing)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    clips, diagrams, info = build(work, args.workers, args.limit)
    out = work / "hf"
    package(clips, diagrams, info, out)

    if args.push:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_large_folder(
            folder_path=str(out), repo_id=REPO_ID, repo_type="dataset"
        )
        local = {p.relative_to(out).as_posix() for p in out.glob("*/*.parquet")}
        stale = [
            f
            for f in api.list_repo_files(REPO_ID, repo_type="dataset")
            if f.endswith(".parquet") and f not in local
        ]
        if stale:
            api.delete_files(
                REPO_ID,
                stale,
                repo_type="dataset",
                commit_message="Remove shards from the previous build",
            )
        print(f"{REPO_ID} revision: {api.dataset_info(REPO_ID).sha}")


if __name__ == "__main__":
    main()
