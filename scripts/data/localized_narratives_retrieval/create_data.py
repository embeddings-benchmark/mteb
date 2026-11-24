"""Build spoken-narration <-> image retrieval tasks from Localized Narratives (Open Images test).

Source
------
Localized Narratives (Pont-Tuset et al., ECCV 2020, arXiv:1912.03098) asked annotators
to describe an image out loud while moving a mouse over the regions they talked about.
Each narrative has the raw voice recording, a manual transcription and a mouse trace.
This script uses the Open Images **test** split:

  annotations  https://storage.googleapis.com/localized-narratives/annotations/
               open_images_test_localized_narratives.jsonl          (CC BY 4.0)
  voice        https://storage.googleapis.com/localized-narratives/voice-recordings/<path>.ogg
  images       https://open-images-dataset.s3.amazonaws.com/test/<image_id>.jpg
  image meta   https://storage.googleapis.com/openimages/2018_04/test/
               test-images-with-rotation.csv                         (per-image CC BY 2.0)

The COCO and Flickr30k subsets of Localized Narratives are not used because those
images already appear in SpokenCOCO / SpeechCoco / Flickr Audio tasks.

Why this is a natural a2i / i2a task
------------------------------------
The recordings are unscripted speech produced while looking at the image: the
narration is the original cross-modal pairing, not a caption read aloud and not a
classification label turned into a query.

Construction
------------
1. Stream the 3.5 GB annotation file once and drop the mouse traces.
2. Keep narratives whose image has Open Images metadata (author, license, rotation)
   and whose transcription and timing look sane (see ``eligible``).
3. Order images by sha1(seed:image_id). If an image has several narratives (0.2% of
   images), keep the one with the smallest sha1(seed:recording) so qrels stay 1:1.
   About 3% of recordings return HTTP 404 from the Localized Narratives bucket, so
   10% spare candidates are fetched in the same order and the first ``--n-images``
   pairs whose audio and image both download are kept.
4. Audio: the 48 kHz stereo Ogg Vorbis recording is decoded and re-encoded to
   16 kHz mono FLAC, untrimmed.
5. Images: the Open Images S3 copy is rotated upright using the metadata ``Rotation``
   column (the S3 files are not pre-rotated) and downscaled to a 1024 px long side
   if larger (4 of ~2.1k candidates). Other images keep their original bytes.
6. One dataset is written, with queries = narrations (audio) and corpus = images.
   LocalizedNarrativesI2ARetrieval loads the same repo and swaps the roles in
   ``dataset_transform``. The image rows carry their Flickr attribution (author,
   license, landing URL).

Usage:
  python scripts/data/localized_narratives_retrieval/create_data.py --work work/ln
  python scripts/data/localized_narratives_retrieval/create_data.py --work work/ln --push
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from datasets import Audio, Dataset, Image, Value
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, create_repo
from PIL import Image as PILImage
from tqdm import tqdm

LN_ROOT = "https://storage.googleapis.com/localized-narratives"
ANNOTATIONS_URL = f"{LN_ROOT}/annotations/open_images_test_localized_narratives.jsonl"
VOICE_URL = f"{LN_ROOT}/voice-recordings/{{path}}"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg"
OI_META_URL = (
    "https://storage.googleapis.com/openimages/2018_04/test/"
    "test-images-with-rotation.csv"
)
SPLIT = "test"
SAMPLE_RATE = 16000
MAX_IMAGE_SIDE = 1024  # the Open Images S3 copies are mostly 1024 px already
SEED = "mteb-localized-narratives-v1"

# eligibility (see ``eligible``)
MIN_WORDS = 5
MIN_SPEECH_S = 2.0
MAX_SPEECH_S = 120.0

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mteb-localized-narratives-builder/1.0"})


def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for _ in range(retries):
        try:
            # stream with a total deadline: a read timeout alone never fires on a
            # connection that trickles bytes, which hung an earlier build.
            deadline = time.monotonic() + 120
            with SESSION.get(url, timeout=(10, 30), stream=True) as r:
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"
                    if r.status_code == 404:
                        break
                    continue
                body = bytearray()
                for chunk in r.iter_content(chunk_size=1 << 16):
                    body += chunk
                    if time.monotonic() > deadline:
                        raise TimeoutError("download exceeded 120 s")
                return bytes(body)
        except Exception as e:  # noqa: BLE001
            last = repr(e)
    raise RuntimeError(f"failed {url}: {last}")


def _h(s: str) -> str:
    return hashlib.sha1(f"{SEED}:{s}".encode()).hexdigest()


# ------------------------------------------------------------ stage 1: index
def _slim(line: bytes) -> dict:
    d = json.loads(line)
    timed = d.get("timed_caption") or []
    return {
        "image_id": d["image_id"],
        "annotator_id": d["annotator_id"],
        "caption": d["caption"],
        "voice_recording": d["voice_recording"],
        "speech_end": max((t["end_time"] for t in timed), default=None),
        "n_timed_words": len(timed),
    }


def index_narratives(work: Path, retries: int = 20) -> list[dict]:
    """Stream the 3.5 GB annotation file once and keep everything except mouse traces.

    A dropped connection resumes with an HTTP Range request from the byte after the
    last complete record, so a long download never has to restart from zero.
    """
    out = work / "narratives.jsonl"
    if out.exists():
        return [json.loads(line) for line in out.open()]
    tmp = out.with_suffix(".part")
    rows: list[dict] = []
    consumed = 0  # bytes up to and including the last fully parsed newline
    total = None
    bar = tqdm(unit="B", unit_scale=True, desc="annotations")
    with tmp.open("w") as fh:
        for attempt in range(retries + 1):
            headers = {"Range": f"bytes={consumed}-"} if consumed else {}
            buf = b""
            try:
                with SESSION.get(
                    ANNOTATIONS_URL, stream=True, timeout=120, headers=headers
                ) as r:
                    r.raise_for_status()
                    if consumed and r.status_code != 206:
                        raise RuntimeError("server ignored the Range request")
                    if total is None:
                        total = int(r.headers.get("content-length", 0))
                        bar.total = total
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        bar.update(len(chunk))
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for line in lines:
                            consumed += len(line) + 1
                            if line.strip():
                                row = _slim(line)
                                rows.append(row)
                                fh.write(json.dumps(row) + "\n")
                if buf.strip():  # final record without a trailing newline
                    row = _slim(buf)
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                break
            except (requests.RequestException, ConnectionError) as e:
                if attempt == retries:
                    raise
                bar.update(-len(buf))
                print(f"\nstream dropped at byte {consumed} ({e!r}); resuming")
    bar.close()
    if total and consumed + len(buf) < total:
        raise RuntimeError(f"annotation stream incomplete: {consumed}/{total} bytes")
    tmp.rename(out)
    return rows


def load_oi_meta(work: Path) -> dict[str, dict]:
    p = work / "test-images-with-rotation.csv"
    if not p.exists():
        p.write_bytes(_get(OI_META_URL))
    with p.open(newline="") as fh:
        return {r["ImageID"]: r for r in csv.DictReader(fh)}


# ----------------------------------------------------------- stage 2: select
def eligible(n: dict, meta: dict[str, dict]) -> bool:
    if n["image_id"] not in meta:
        return False
    if len(n["caption"].split()) < MIN_WORDS:
        return False
    end = n["speech_end"]
    return end is not None and MIN_SPEECH_S <= end <= MAX_SPEECH_S


def select(rows: list[dict], meta: dict[str, dict], n_candidates: int) -> list[dict]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for n in rows:
        if eligible(n, meta):
            by_image[n["image_id"]].append(n)
    multi = sum(len(v) > 1 for v in by_image.values())
    print(
        f"{len(rows)} narratives -> {sum(map(len, by_image.values()))} eligible "
        f"on {len(by_image)} images ({multi} images with >1 eligible narrative)"
    )
    images = sorted(by_image, key=_h)[:n_candidates]
    chosen = []
    for iid in images:
        n = min(by_image[iid], key=lambda x: _h(x["voice_recording"]))
        chosen.append(n)
    return chosen


# ----------------------------------------------------------- stage 3: media
def fetch_audio(n: dict, work: Path) -> Path:
    out = work / "audio" / (Path(n["voice_recording"]).stem + ".flac")
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = _get(VOICE_URL.format(path=n["voice_recording"]))
    part = out.with_suffix(".part.flac")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "flac", str(part),
        ],
        input=raw, check=True, capture_output=True, timeout=120,
    )  # fmt: skip
    part.rename(out)
    return out


def fetch_image(n: dict, meta: dict[str, dict], work: Path) -> Path:
    out = work / "images" / f"{n['image_id']}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = _get(IMAGE_URL.format(image_id=n["image_id"]))
    im = PILImage.open(io.BytesIO(raw)).convert("RGB")
    rot = float(meta[n["image_id"]]["Rotation"] or 0.0)
    oversized = max(im.size) > MAX_IMAGE_SIDE  # a handful of S3 copies are full-size
    if not rot and not oversized:
        out.write_bytes(raw)  # untouched original bytes
        return out
    if rot:
        # Rotation is the counter-clockwise angle that makes the S3 copy upright.
        im = im.rotate(rot, expand=True)
    if oversized:
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), PILImage.LANCZOS)
    part = out.with_suffix(".part.jpg")
    im.save(part, quality=95)
    part.rename(out)
    return out


def audio_seconds(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(p)],
        check=True, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    return float(r.stdout.strip())


def fetch_all(chosen: list[dict], meta: dict[str, dict], work: Path, workers: int):
    ok, bad = [], 0

    def one(n):
        a = fetch_audio(n, work)
        i = fetch_image(n, meta, work)
        return {
            **n,
            "audio_path": str(a),
            "image_path": str(i),
            "audio_s": audio_seconds(a),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, n): n for n in chosen}
        for f in tqdm(as_completed(futs), total=len(futs), desc="media"):
            try:
                ok.append(f.result())
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"  FAILED {futs[f]['voice_recording']}: {e}")
    if bad:
        print(
            f"{bad} narratives dropped because their media failed to download or decode"
        )
    order = {n["voice_recording"]: k for k, n in enumerate(chosen)}
    return sorted(ok, key=lambda n: order[n["voice_recording"]])


# --------------------------------------------------------- stage 4: package
def audio_id(n: dict) -> str:
    return Path(n["voice_recording"]).stem  # open_images_test_<image_id>_<annotator>


def build_datasets(items: list[dict], meta: dict[str, dict]):
    audio = (
        Dataset.from_dict(
            {
                "_id": [audio_id(n) for n in items],
                "audio": [n["audio_path"] for n in items],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    )
    m = [meta[n["image_id"]] for n in items]
    images = (
        Dataset.from_dict(
            {
                "_id": [n["image_id"] for n in items],
                "image": [n["image_path"] for n in items],
                "author": [r["Author"] for r in m],
                "license": [r["License"] for r in m],
                "source_url": [r["OriginalLandingURL"] for r in m],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("image", Image())
    )
    pairs = [(audio_id(n), n["image_id"]) for n in items]
    return audio, images, pairs


def qrels(pairs: list[tuple[str, str]]) -> Dataset:
    return (
        Dataset.from_dict(
            {
                "query-id": [q for q, _ in pairs],
                "corpus-id": [c for _, c in pairs],
                "score": [1] * len(pairs),
            }
        )
        .cast_column("query-id", Value("string"))
        .cast_column("corpus-id", Value("string"))
        .cast_column("score", Value("int32"))
    )


CARD = """---
license: cc-by-4.0
language:
- en
pretty_name: {pretty}
size_categories:
- 1K<n<10K
configs:
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

# {pretty}

{direction} retrieval built from the Open Images **test** subset of
[Localized Narratives](https://google.github.io/localized-narratives/) for
[MTEB](https://github.com/embeddings-benchmark/mteb) (`{task}`).

Annotators described each image out loud while looking at it. Each query is matched with
the one item from the same narrative ({n} pairs, 1:1 qrels). Images were sampled with
a fixed seed. When an image had several narratives, one was chosen by hash.

| config | split | rows | contents |
|---|---|---|---|
| `queries` | `test` | {n} | {q_desc} |
| `corpus` | `test` | {n} | {c_desc} |
| `default` | `test` | {n} | `query-id`, `corpus-id`, `score` (1) |

Audio is the original recording, decoded from 48 kHz stereo Ogg Vorbis and re-encoded as
16 kHz mono FLAC, untrimmed. Images are the Open Images S3 copies (long side at most
1024 px), rotated upright using the Open Images `Rotation` metadata. Build script:
`scripts/data/localized_narratives_retrieval/create_data.py` in MTEB.

## License and attribution

- Localized Narratives annotations: CC BY 4.0, Google LLC. The voice recordings are
  distributed with the annotations and no separate license is stated for them.
- Images: each image is CC BY 2.0. Its Flickr author, license and source page are in
  the `author`, `license` and `source_url` columns of the image config.
- Changes: audio resampled and re-encoded; images rotated where the Open Images
  metadata says so and downscaled to 1024 px where larger; subset selection.

## Citation

```bibtex
@inproceedings{{pont-tuset2020localized,
  author = {{Pont-Tuset, Jordi and Uijlings, Jasper and Changpinyo, Soravit and Soricut, Radu and Ferrari, Vittorio}},
  booktitle = {{ECCV}},
  title = {{Connecting Vision and Language with Localized Narratives}},
  year = {{2020}},
}}
```
"""
AUDIO_DESC = "`_id`, `audio` (spoken narration, 16 kHz mono)"
IMAGE_DESC = "`_id`, `image`, `author`, `license`, `source_url`"


def write_and_push(name, queries, corpus, qr, card, out: Path, repo_id, push, private):
    d = out / name
    for ds, sub in ((corpus, "corpus"), (queries, "queries"), (qr, "data")):
        (d / sub).mkdir(parents=True, exist_ok=True)
        # embed media bytes so the local copy is self-contained (push_to_hub does this too)
        embedded = ds.with_format("arrow").map(embed_table_storage, batched=True)
        embedded.with_format(None).to_parquet(
            d / sub / f"{SPLIT}-00000-of-00001.parquet"
        )
    (d / "README.md").write_text(card)
    print(f"wrote {d}")
    if not push:
        return
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    for ds, cfg in ((corpus, "corpus"), (queries, "queries"), (qr, "default")):
        ds.push_to_hub(repo_id, config_name=cfg, split=SPLIT)
        print(f"  pushed {repo_id} config {cfg!r}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", commit_message="Add dataset card",
    )  # fmt: skip
    print(f"  {repo_id} revision: {api.dataset_info(repo_id).sha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/localized_narratives")
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--a2i-repo", default="mteb/LocalizedNarrativesA2I")
    ap.add_argument("--stats-only", action="store_true", help="stop after selection")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    rows = index_narratives(work)
    meta = load_oi_meta(work)
    per_image = Counter(n["image_id"] for n in rows)
    ends = np.array([n["speech_end"] for n in rows if n["speech_end"] is not None])
    print(
        f"{len(rows)} narratives on {len(per_image)} images; "
        f"narratives per image: {dict(Counter(per_image.values()))}"
    )
    print(
        "speech_end s: "
        + ", ".join(f"p{p}={np.percentile(ends, p):.1f}" for p in (1, 10, 50, 90, 99))
        + f", missing={sum(n['speech_end'] is None for n in rows)}"
    )
    # ~3% of recordings 404 in the LN bucket, so fetch spare candidates in the same
    # hash order and keep the first --n-images whose media all download.
    chosen = select(rows, meta, int(args.n_images * 1.1))
    if args.stats_only:
        return

    items = fetch_all(chosen, meta, work, args.workers)
    if len(items) < args.n_images:
        raise RuntimeError(f"only {len(items)} usable pairs, wanted {args.n_images}")
    items = items[: args.n_images]
    secs = np.array([n["audio_s"] for n in items])
    print(
        f"{len(items)} pairs; audio seconds total={secs.sum() / 3600:.2f} h, "
        + ", ".join(f"p{p}={np.percentile(secs, p):.1f}" for p in (0, 50, 90, 100))
    )

    audio, images, pairs = build_datasets(items, meta)
    assert len(set(audio["_id"])) == len(audio), "duplicate audio _id"
    assert len(set(images["_id"])) == len(images), "duplicate image _id"
    n = len(pairs)
    out = work / "hf"
    write_and_push(
        "a2i", audio, images, qrels(pairs),
        CARD.format(pretty="Localized Narratives A2I", direction="Spoken narration to image",
                    task="LocalizedNarrativesA2IRetrieval", n=n,
                    q_desc=AUDIO_DESC, c_desc=IMAGE_DESC),
        out, args.a2i_repo, args.push, args.private,
    )  # fmt: skip


if __name__ == "__main__":
    main()
