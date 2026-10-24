"""Package Amazon Berkeley Objects (ABO) into an MTEB image-to-video retrieval task.

Source: https://amazon-berkeley-objects.s3.amazonaws.com (CC BY 4.0).
Only the ``spins/`` (turntable photography) and ``images/`` (catalog photography)
directories plus their metadata are used.

Why ABO is natively cross-modal
-------------------------------
ABO ships, for the same product, two independently captured assets:

  * a 360-degree turntable "spin" -- a real frame sequence shot in a rig, and
  * catalog photographs -- separate stills shot at a different time, in a
    different place, under different lighting.

A query is therefore never a frame of its own positive video, and no crop,
re-encode or temporal-neighbour relationship links the two. Frame leakage is
structurally impossible rather than filtered out after the fact, which is the
failure mode that saturates i2v tasks built by slicing a query frame out of the
corpus clip.

Construction
------------
corpus  = one video per product, encoded from that product's spin. Frames are
          selected by ``azimuth % 3 == 0``, which yields exactly 24 frames at
          15-degree steps for every spin in ABO: the 8,116 dense spins store
          azimuths 0..71 and the 93 sparse ones store exactly {0,3,...,69}, so a
          single rule makes the corpus homogeneous without dropping any sequence.
queries = one catalog photograph per product, drawn from the "context" bucket
          (see below) -- the product photographed in a room or scene.
qrels   = the query for product P is relevant to P's video only, score 1.

Catalog images are bucketed, and only "context" is eligible as a query:

  placeholder   pHash shared with another product (boilerplate, size charts,
                brand cards). Excluded -- these are not photographs of the item.
  swatch/crop   fails a zero-shot category gate (flat fabric fills, macro crops
                of a caster or a screw, dimension diagrams). Excluded.
  near-dup      pHash within ``--min-phash-gap`` of any frame of the product's
                own spin. Excluded, belt-and-braces against leakage even though
                the capture sessions are independent.
  studio        product alone on a white sweep (white pixel fraction >= 0.5).
                Not used: too close to the spin's own studio conditions, which
                makes retrieval trivial.
  context       everything else -- the product in situ. THIS is the query set.

Selecting queries by a semantic category ("a lifestyle photograph of the
product") rather than by distance from the answer keeps every query a genuine,
answerable depiction of the item. The difficulty comes from corpus size and from
the domain gap between a styled room photo and a turntable render, not from
degrading the query.

Product types are restricted to five volumetric home-goods categories. Flat
goods (RUG, WALL_ART) are excluded because a turntable rotation of a flat object
is close to degenerate.

Usage:
  uv run python scripts/data/abo_i2v_retrieval/create_data.py --work work
  uv run python scripts/data/abo_i2v_retrieval/create_data.py --work work \
      --repo-id hubxrt/ABO-I2V --push
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets import Dataset, Image, Value, Video
from huggingface_hub import HfApi, create_repo
from PIL import Image as PILImage
from tqdm import tqdm

S3 = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTING_SHARDS = "0123456789abcdef"  # 16 shards, hex-suffixed
SPLIT = "test"

PRODUCT_TYPES = (
    "CHAIR",
    "SOFA",
    "TABLE",
    "HOME_FURNITURE_AND_DECOR",
    "LAMP",
)
TYPE_LABEL = {
    "CHAIR": "chair",
    "SOFA": "sofa",
    "TABLE": "table",
    "HOME_FURNITURE_AND_DECOR": "piece of home furniture or decor",
    "LAMP": "lamp",
}
JUNK_PROMPTS = (
    "a flat fabric swatch, a solid color texture",
    "a blank placeholder image with text",
    "an extreme close-up of a small furniture part",
    "a diagram with measurements and dimensions",
)

# frame selection: 24 frames at 15-degree steps, uniform across all ABO spins
AZIMUTH_STEP = 3
N_FRAMES = 24
# encode
LONG_SIDE = 384
CRF = 23
FPS = 12
# query eligibility
MIN_CATALOG_SIDE = 256  # on the ORIGINAL, a source-quality gate
MAX_CATALOG_PER_PRODUCT = 10
PROBE_FRAMES = 12  # spin frames used for the near-dup pHash check

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mteb-abo-i2v-builder/1.0"})


# --------------------------------------------------------------------- helpers
def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for _ in range(retries):
        try:
            r = SESSION.get(url, timeout=90)
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
    raise RuntimeError(f"failed {url}: {last}")


def _cached(path: Path, url: str) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(url))
    return path


def _first(v):
    """ABO metadata stores many fields as [{'value': ...}, ...]."""
    if isinstance(v, list):
        return _first(v[0]) if v else None
    if isinstance(v, dict):
        return v.get("value")
    return v


def _csv_gz(work: Path, name: str) -> list[dict]:
    p = work / f"{name}.csv"
    if not p.exists():
        raw = _get(f"{S3}/{name}/metadata/{name}.csv.gz")
        p.write_bytes(gzip.decompress(raw))
    with p.open() as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------ stage 1: metadata
def build_products(work: Path) -> list[dict]:
    cache = work / "products.json"
    if cache.exists():
        return json.loads(cache.read_text())

    print("reading spins metadata")
    spins: dict[str, list[dict]] = defaultdict(list)
    for r in _csv_gz(work, "spins"):
        if int(r["azimuth"]) % AZIMUTH_STEP == 0:
            spins[r["spin_id"]].append(r)
    for v in spins.values():
        v.sort(key=lambda r: int(r["azimuth"]))
    spins = {k: v for k, v in spins.items() if len(v) == N_FRAMES}
    print(
        f"  {len(spins)} spins yield exactly {N_FRAMES} frames at {AZIMUTH_STEP}-azimuth steps"
    )

    print("reading images metadata")
    images: dict[str, dict] = {}
    for r in _csv_gz(work, "images"):
        if min(int(r["width"]), int(r["height"])) >= MIN_CATALOG_SIDE:
            images[r["image_id"]] = r
    print(f"  {len(images)} catalog images with min side >= {MIN_CATALOG_SIDE}px")

    print(f"reading {len(LISTING_SHARDS)} listings shards")
    wanted = set(PRODUCT_TYPES)
    by_spin: dict[str, list[dict]] = defaultdict(list)
    for i in tqdm(LISTING_SHARDS, desc="listings"):
        p = work / f"listings_{i}.json"
        if not p.exists():
            p.write_bytes(
                gzip.decompress(_get(f"{S3}/listings/metadata/listings_{i}.json.gz"))
            )
        for line in p.open():
            d = json.loads(line)
            sid = d.get("spin_id")
            if not sid or sid not in spins:
                continue
            ptype = _first(d.get("product_type"))
            if ptype not in wanted:
                continue
            cat = []
            for iid in [d.get("main_image_id")] + (d.get("other_image_id") or []):
                if iid and iid in images and iid not in cat:
                    cat.append(iid)
            if not cat:
                continue
            by_spin[sid].append(
                {
                    "item_id": d["item_id"],
                    "product_type": ptype,
                    "spin_id": sid,
                    "catalog": cat[:MAX_CATALOG_PER_PRODUCT],
                }
            )

    # dedup on spin_id: a handful of listings share a sequence, which would make
    # two queries share one correct document.
    products, dropped = [], 0
    for sid, group in sorted(by_spin.items()):
        if len(group) > 1:
            dropped += len(group) - 1
        products.append(sorted(group, key=lambda d: d["item_id"])[0])
    products.sort(key=lambda d: d["spin_id"])
    print(
        f"{len(products)} products after spin_id dedup ({dropped} duplicate listings dropped)"
    )

    counts = defaultdict(int)
    for p in products:
        counts[p["product_type"]] += 1
    for t in PRODUCT_TYPES:
        print(f"  {t:28s}{counts[t]}")

    for p in products:
        p["frames"] = [
            {"image_id": r["image_id"], "path": r["path"]} for r in spins[p["spin_id"]]
        ]
        p["catalog_paths"] = {iid: images[iid]["path"] for iid in p["catalog"]}
    cache.write_text(json.dumps(products))
    return products


# ------------------------------------------------- stage 2: bucket catalog images
def _phash(p: Path):
    try:
        return imagehash.phash(PILImage.open(p).convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


def _white_frac(p: Path):
    try:
        im = PILImage.open(p).convert("L")
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(im.resize((128, 128), PILImage.BILINEAR), dtype=np.float32)
    return float((a > 240).mean())


def _tensor_of(v):
    if torch.is_tensor(v):
        return v
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        c = getattr(v, attr, None)
        if c is not None:
            return c
    return v.last_hidden_state.mean(1)


def bucket_catalog(
    work: Path, products: list[dict], model_name: str, min_phash_gap: int, workers: int
) -> list[dict]:
    cache = work / "buckets.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        # guard: a cache built from a different (e.g. --limit'ed) product set is
        # not reusable, and silently reusing it would truncate the task.
        if blob.get("n_products") == len(products):
            return blob["recs"]
        print(
            f"buckets cache is for {blob.get('n_products')} products, "
            f"now {len(products)} -- rebuilding"
        )

    cdir = work / "cache"
    cdir.mkdir(parents=True, exist_ok=True)

    # download probe spin frames + all candidate catalog images
    jobs = {}
    for p in products:
        idx = sorted(
            {
                round(i * (N_FRAMES - 1) / (PROBE_FRAMES - 1))
                for i in range(PROBE_FRAMES)
            }
        )
        p["probe_frames"] = [p["frames"][i]["image_id"] for i in idx]
        for i in idx:
            f = p["frames"][i]
            jobs[f["image_id"]] = f"{S3}/spins/original/{f['path']}"
        for iid, path in p["catalog_paths"].items():
            jobs[iid] = f"{S3}/images/small/{path}"
    todo = [(i, u) for i, u in jobs.items() if not (cdir / f"{i}.img").exists()]
    print(f"{len(jobs)} probe images, {len(todo)} to download")
    if todo:

        def one(t):
            iid, url = t
            try:
                (cdir / f"{iid}.img").write_bytes(_get(url))
            except Exception:  # noqa: BLE001
                pass

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(
                tqdm(
                    as_completed([ex.submit(one, t) for t in todo]),
                    total=len(todo),
                    desc="probe download",
                )
            )

    # pHash everything once
    print("hashing")
    ph: dict[str, object] = {}
    for iid in tqdm(jobs, desc="phash"):
        h = _phash(cdir / f"{iid}.img")
        if h is not None:
            ph[iid] = h

    # a catalog pHash owned by more than one product is boilerplate
    owners: dict[str, set[str]] = defaultdict(set)
    for p in products:
        for iid in p["catalog"]:
            if iid in ph:
                owners[str(ph[iid])].add(p["spin_id"])
    shared = {h for h, o in owners.items() if len(o) > 1}
    print(f"{len(shared)} pHash values shared across products -> placeholder")

    # zero-shot category gate
    from transformers import AutoModel, AutoProcessor

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading {model_name} on {dev} (construction gate, not evaluation)")
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()

    texts, slot = [], {}
    for t in PRODUCT_TYPES:
        lab = TYPE_LABEL[t]
        slot[t] = (len(texts), len(texts) + 2)
        texts += [f"a photo of a {lab}", f"a {lab} in a furnished room"]
    junk0 = len(texts)
    texts += list(JUNK_PROMPTS)
    with torch.no_grad():
        ti = proc(text=texts, padding="max_length", return_tensors="pt").to(dev)
        T = F.normalize(_tensor_of(model.get_text_features(**ti)).float(), dim=-1).cpu()

    recs = []
    for p in products:
        fh = [ph[i] for i in p["probe_frames"] if i in ph]
        if not fh:
            continue
        for iid in p["catalog"]:
            if iid not in ph:
                continue
            w = _white_frac(cdir / f"{iid}.img")
            if w is None:
                continue
            recs.append(
                {
                    "spin_id": p["spin_id"],
                    "product_type": p["product_type"],
                    "image_id": iid,
                    "white": float(w),
                    "shared": str(ph[iid]) in shared,
                    "gap": int(min(ph[iid] - x for x in fh)),
                }
            )

    print(f"gating {len(recs)} catalog images")
    with torch.no_grad():
        sims = []
        bs = 32
        for i in tqdm(range(0, len(recs), bs), desc="embed"):
            ims = [
                PILImage.open(cdir / f"{r['image_id']}.img").convert("RGB")
                for r in recs[i : i + bs]
            ]
            inp = proc(images=ims, return_tensors="pt").to(dev)
            E = F.normalize(
                _tensor_of(model.get_image_features(**inp)).float(), dim=-1
            ).cpu()
            sims.append(E @ T.T)
        S = torch.cat(sims)

    for n, r in enumerate(recs):
        a, b = slot[r["product_type"]]
        r["is_product"] = bool(S[n, a:b].max() > S[n, junk0:].max())
        if r["shared"]:
            r["bucket"] = "placeholder"
        elif not r["is_product"]:
            r["bucket"] = "swatch/crop"
        elif r["gap"] < min_phash_gap:
            r["bucket"] = "near-dup"
        else:
            r["bucket"] = "context" if r["white"] < 0.5 else "studio"

    tally = defaultdict(int)
    for r in recs:
        tally[r["bucket"]] += 1
    print("\ncatalog image buckets:")
    for k in ("context", "studio", "swatch/crop", "placeholder", "near-dup"):
        print(f"  {k:14s}{tally[k]}")
    cache.write_text(json.dumps({"n_products": len(products), "recs": recs}))
    return recs


# ------------------------------------------------------- stage 3: select queries
def select(products: list[dict], recs: list[dict], target: int):
    by_spin = {p["spin_id"]: p for p in products}
    ctx: dict[str, dict] = {}
    for r in recs:
        if r["bucket"] != "context":
            continue
        cur = ctx.get(r["spin_id"])
        # least-white context image = most scene-like
        if cur is None or r["white"] < cur["white"]:
            ctx[r["spin_id"]] = r
    print(
        f"\n{len(ctx)}/{len(products)} products have a context query "
        f"({100 * len(ctx) / len(products):.1f}%)"
    )
    chosen = sorted(ctx.values(), key=lambda r: r["spin_id"])[:target]
    return [
        {
            **by_spin[r["spin_id"]],
            "query_image_id": r["image_id"],
            "query_white": r["white"],
        }
        for r in chosen
    ]


# --------------------------------------------------------- stage 4: encode video
def encode_one(p: dict, work: Path) -> Path | None:
    out = work / "videos" / f"{p['spin_id']}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = work / "tmp_frames" / p["spin_id"]
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for n, f in enumerate(p["frames"]):
            fp = tmp / f"{n:03d}.jpg"
            if not fp.exists() or fp.stat().st_size == 0:
                fp.write_bytes(_get(f"{S3}/spins/original/{f['path']}"))
        part = out.with_suffix(".part.mp4")
        vf = (
            f"scale=w={LONG_SIDE}:h={LONG_SIDE}:force_original_aspect_ratio=decrease"
            ":force_divisible_by=2:in_range=full:out_range=tv,format=yuv420p"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(tmp / "%03d.jpg"),
            "-vf",
            vf,
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-movflags",
            "+faststart",
            str(part),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        # validate decode of every output
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,width,height,pix_fmt",
                "-of",
                "default=nw=1:nk=1",
                str(part),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        vals = pr.stdout.split()
        nb = int(vals[-1] if vals[-1].isdigit() else vals[0])
        if nb != N_FRAMES:
            raise RuntimeError(
                f"{p['spin_id']}: decoded {nb} frames, expected {N_FRAMES}"
            )
        part.rename(out)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def encode_all(selected: list[dict], work: Path, workers: int):
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(encode_one, p, work): p for p in selected}
        for f in tqdm(as_completed(futs), total=len(futs), desc="encode"):
            p = futs[f]
            try:
                r = f.result()
                (ok if r else bad).append(p)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {p['spin_id']}: {e}", file=sys.stderr)
                bad.append(p)
    if bad:
        print(f"{len(bad)} spins failed to encode/validate and are excluded")
    return [p for p in selected if p in ok]


# -------------------------------------------------------------- stage 5: package
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--repo-id", default="hubxrt/ABO-I2V")
    ap.add_argument("--target", type=int, default=3500)
    ap.add_argument("--min-phash-gap", type=int, default=10)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap products")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    products = build_products(work)
    if args.limit:
        products = products[: args.limit]
    recs = bucket_catalog(work, products, args.model, args.min_phash_gap, args.workers)
    selected = select(products, recs, args.target)
    print(f"{len(selected)} products selected for the task")

    selected = encode_all(selected, work, args.workers)

    # materialise query images with a real extension
    qdir = work / "queries"
    qdir.mkdir(parents=True, exist_ok=True)
    for p in selected:
        dst = qdir / f"{p['query_image_id']}.jpg"
        if not dst.exists():
            shutil.copyfile(work / "cache" / f"{p['query_image_id']}.img", dst)

    cids = [p["spin_id"] for p in selected]
    qids = [p["query_image_id"] for p in selected]
    assert len(set(cids)) == len(cids), "duplicate corpus _id"
    assert len(set(qids)) == len(qids), "duplicate query _id"

    corpus = (
        Dataset.from_dict(
            {
                "_id": cids,
                "video": [str(work / "videos" / f"{c}.mp4") for c in cids],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("video", Video())
    )
    queries = (
        Dataset.from_dict(
            {
                "_id": qids,
                "image": [str(qdir / f"{q}.jpg") for q in qids],
            }
        )
        .cast_column("_id", Value("string"))
        .cast_column("image", Image())
    )
    qrels = (
        Dataset.from_dict(
            {
                "query-id": qids,
                "corpus-id": cids,
                "score": [1] * len(cids),
            }
        )
        .cast_column("query-id", Value("string"))
        .cast_column("corpus-id", Value("string"))
        .cast_column("score", Value("int32"))
    )

    vid_bytes = sum((work / "videos" / f"{c}.mp4").stat().st_size for c in cids)
    print(f"\ncorpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")
    print(
        f"video total {vid_bytes / 1e6:.1f} MB  mean {vid_bytes / len(cids) / 1024:.1f} KB"
    )
    print(f"corpus features : {corpus.features}")
    print(f"queries features: {queries.features}")
    print(f"qrels features  : {qrels.features}")
    tally = defaultdict(int)
    for p in selected:
        tally[p["product_type"]] += 1
    print(
        "per product type: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    )

    if not args.push:
        print("\n--push not set; nothing uploaded.")
        return

    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    for ds, cfg in ((corpus, "corpus"), (queries, "queries"), (qrels, "default")):
        ds.push_to_hub(args.repo_id, config_name=cfg, split=SPLIT)
        print(f"pushed config {cfg!r}")
    sha = HfApi().dataset_info(args.repo_id).sha
    print(f"\nrevision: {sha}")


if __name__ == "__main__":
    main()
