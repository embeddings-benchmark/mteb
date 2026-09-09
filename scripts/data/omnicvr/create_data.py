"""Build the OmniCVR-mini and OmniCVR-mini-hard-negatives datasets.

Both variants use the exact same deterministic 500-query stratified sample
of the full `mteb/OmniCVR` benchmark (5,000 queries), split by the dataset's
`category` field:

    audio-center: 100 / 1000
    visual-center: 114 / 1141
    integrated:    286 / 2859

- `mini`: keeps each of the 500 queries' original 2,000-candidate
  `top_ranked` gallery unchanged. The corpus is reduced to the union of
  candidate videos actually referenced by those galleries (~14.4k videos),
  not the full ~16.3k-video corpus -- this is a pure dedup, it never removes
  a candidate any gallery actually points to.
- `mini-hard-negatives`: same 500 queries, but each 2,000-candidate gallery
  is reduced to 250 (249 hard negatives + the positive) via XCLIP-based
  mining (see "Hard negative mining" below). The mined result is NOT
  checked into this repo -- generated candidate data belongs in the HF
  dataset, not in git. Pass it explicitly via `--mined-top-ranked-path`, or
  regenerate it with `--remine`.

Hard negative mining
---------------------
Per Isaac's clarification on PR #5036 ("Xclip is only used to sample the
corpus, which as I understand is video only"), the mining uses ONLY corpus
(candidate) videos with XCLIP -- it never encodes the query's source video
or modification instruction. This is an implementation decision made to
satisfy that constraint (the positive/target is a corpus entry, not a query
field, so anchoring on it never requires XCLIP to see the query side); it
is not a literal requirement Isaac stated.

  1. For each of the 500 queries, take its qrels positive/target video
     (a corpus entry, not a query field) as the video-only anchor.
  2. Encode every video referenced by any of the 500 galleries with
     `microsoft/xclip-base-patch16` (video-only, `get_video_features`, 8
     uniformly sampled frames, L2-normalized). Each unique video is encoded
     exactly once and cached, even though it may appear in many galleries.
  3. Rank the other 1,999 candidates in that query's original gallery by
     cosine similarity to the positive's embedding, descending.
  4. Keep the 249 most similar (hardest negatives) + the positive.
  5. Order: positive first, then the 249 negatives by descending similarity.

`mine_hard_negatives()` reproduces this exactly and is resumable: it
encodes into a local embedding cache directory (one `.npy` per video id),
skips ids that already have a cached embedding, checkpoints after every
chunk, and refreshes each chunk's signed video URLs immediately before
downloading that chunk (not all up front) -- signed URLs from the HF
datasets-server expire (~1hr TTL); refreshing right before use avoids the
mass-403 failure mode a bulk-refresh-then-consume approach hits partway
through a multi-hour run. It is NOT invoked by the default CLI flow --
encoding ~14.4k videos is an expensive GPU job that should be run
deliberately with `--remine`, not as a side effect of building the dataset.

Usage:
    # mini: no mining involved
    python scripts/data/omnicvr/create_data.py --variant mini

    # mini-hard-negatives: use an already-mined result
    python scripts/data/omnicvr/create_data.py --variant mini-hard-negatives \\
        --mined-top-ranked-path /path/to/mined_top_ranked.json

    # mini-hard-negatives: mine from scratch (expensive GPU job, resumable)
    python scripts/data/omnicvr/create_data.py --variant mini-hard-negatives \\
        --remine --embedding-cache-dir /path/to/cache

    # validate only, no video downloads / dataset assembly
    python scripts/data/omnicvr/create_data.py --variant mini --validate-only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Sequence, Value, Video

SOURCE_REPO = "mteb/OmniCVR"
SOURCE_REVISION = "e0c1031c52fff76113b5917f05b1589ad3f0c61a"

SEED = 42
CATEGORY_TARGETS = {
    "audio-center": 100,
    "visual-center": 114,
    "integrated": 286,
}
CATEGORY_VERIFIED_TOTALS = {
    "audio-center": 1000,
    "visual-center": 1141,
    "integrated": 2859,
}
NUM_QUERIES = 500
FULL_GALLERY_SIZE = 2000
REDUCED_GALLERY_SIZE = 250

XCLIP_MODEL_NAME = "microsoft/xclip-base-patch16"
XCLIP_NUM_FRAMES = 8
MINING_CHUNK_SIZE = 384  # refresh-then-immediately-consume unit
MINING_BATCH_SIZE = 32  # XCLIP forward-pass batch size
MINING_MAX_RETRY_ROUNDS = 3  # bounded retries over whatever is still missing
DOWNLOAD_WORKERS = 16
DECODE_WORKERS = 8


# --------------------------------------------------------------------------
# Stage 1: deterministic 500-query stratified sample (cheap, metadata only)
# --------------------------------------------------------------------------


def load_query_categories(max_retries: int = 5) -> dict[str, str]:
    """id -> category for all 5,000 queries in the full OmniCVR dataset.

    Uses the lightweight datasets-server /rows API (paginated, metadata +
    signed asset URLs only -- never video/audio bytes) instead of
    load_dataset() on the full queries config, which would otherwise
    download/decode ~11.5GB of video+audio just to read two scalar
    columns. Retries transient 5xx/timeout errors per page with bounded
    attempts; raises if a page never succeeds (deterministic failure, no
    silent partial result). No scratchpad/instance-specific paths.
    """
    import urllib.request

    total_rows = 5000
    base = "https://datasets-server.huggingface.co/rows"
    tmpl = (
        "dataset=mteb%2FOmniCVR&config=queries&split=queries&offset={offset}&length=100"
    )
    id_to_category: dict[str, str] = {}
    for offset in range(0, total_rows, 100):
        url = f"{base}?{tmpl.format(offset=offset)}"
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
                break
            except Exception as e:  # noqa: BLE001
                _log(
                    f"  query-category fetch retry offset={offset} attempt={attempt}: {e}"
                )
                time.sleep(1)
        else:
            raise RuntimeError(
                f"failed to fetch query categories at offset {offset} "
                f"after {max_retries} attempts"
            )
        for r in data["rows"]:
            row = r["row"]
            id_to_category[row["id"]] = row["category"]

    if len(id_to_category) != total_rows:
        raise RuntimeError(
            f"expected {total_rows} query categories, got {len(id_to_category)}"
        )
    return id_to_category


def sample_500_queries(id_to_category: dict[str, str]) -> dict[str, list[str]]:
    """Reproduce the exact deterministic stratified sample.

    Method: per-category pool sorted by id, `random.Random(SEED).shuffle`,
    take the first `k`, re-sort. Same method + same seed => same 500 ids,
    every time, on any machine.
    """
    by_category: dict[str, list[str]] = {}
    for qid, cat in id_to_category.items():
        by_category.setdefault(cat, []).append(qid)

    for cat, expected_total in CATEGORY_VERIFIED_TOTALS.items():
        actual = len(by_category.get(cat, []))
        if actual != expected_total:
            raise ValueError(
                f"Category {cat!r} has {actual} queries, expected "
                f"{expected_total}. The source dataset may have changed; "
                "do not silently resample -- investigate first."
            )

    sample: dict[str, list[str]] = {}
    for cat, k in CATEGORY_TARGETS.items():
        pool = sorted(by_category[cat])
        rng = random.Random(SEED)
        rng.shuffle(pool)
        sample[cat] = sorted(pool[:k])

    total = sum(len(v) for v in sample.values())
    if total != NUM_QUERIES:
        raise ValueError(f"Expected {NUM_QUERIES} sampled queries, got {total}")
    return sample


def all_sampled_ids(sample: dict[str, list[str]]) -> list[str]:
    return sorted(qid for ids in sample.values() for qid in ids)


# --------------------------------------------------------------------------
# Stage 2: load qrels / top_ranked / queries / corpus for the sampled ids
# --------------------------------------------------------------------------


def load_qrels_for(query_ids: set[str], num_proc: int | None = None) -> dict[str, str]:
    """query-id -> positive corpus-id, for the given queries."""
    from datasets import load_dataset

    qrels = load_dataset(
        SOURCE_REPO, "qrels", split="test", revision=SOURCE_REVISION, num_proc=num_proc
    )
    return {
        row["query-id"]: row["corpus-id"]
        for row in qrels
        if row["query-id"] in query_ids
    }


def load_top_ranked_for(
    query_ids: set[str], num_proc: int | None = None
) -> dict[str, list[str]]:
    """query-id -> original 2,000-candidate gallery, for the given queries."""
    from datasets import load_dataset

    top_ranked = load_dataset(
        SOURCE_REPO,
        "top_ranked",
        split="test",
        revision=SOURCE_REVISION,
        num_proc=num_proc,
    )
    return {
        row["query-id"]: row["corpus-ids"]
        for row in top_ranked
        if row["query-id"] in query_ids
    }


def load_mined_top_ranked(path: str | Path) -> dict[str, list[str]]:
    """Load a precomputed mining result. Not checked into this repo -- pass
    the path explicitly (e.g. a local copy, or one exported from the HF
    dataset's own `top_ranked` config)."""
    with Path(path).open() as f:
        return json.load(f)


def unique_corpus(top_ranked: dict[str, list[str]]) -> set[str]:
    return {cid for cands in top_ranked.values() for cid in cands}


# --------------------------------------------------------------------------
# Hard negative mining (NOT run by default -- see module docstring).
# Resumable: encode-once, chunk-checkpointed, refresh-close-to-use, bounded
# retries. This reproduces the pipeline validated in the PR discussion.
# --------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sample_frames(path: str, num_frames: int = XCLIP_NUM_FRAMES) -> list[Any]:
    import av
    import numpy as np

    container = av.open(path)
    total = container.streams.video[0].frames
    idxs = set(np.linspace(0, max(total - 1, 0), num_frames).astype(int).tolist())
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in idxs:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])
    container.close()
    return frames


def _fetch_video_urls(corpus_ids: set[str] | None = None) -> dict[str, str]:
    """Fetch current (freshly signed) video URLs for corpus ids, via the
    lightweight datasets-server rows API (returns signed asset URLs, not
    video bytes -- cheap even for the full ~16.3k-row corpus).

    If `corpus_ids` is given, still scans the whole corpus (there is no
    server-side filter by id) but only keeps entries that are needed --
    call this once per mining chunk, right before downloading that chunk,
    so URLs are always fresh when consumed.
    """
    import urllib.request

    base = "https://datasets-server.huggingface.co/rows"
    tmpl = (
        "dataset=mteb%2FOmniCVR&config=corpus&split=corpus&offset={offset}&length=100"
    )
    urls: dict[str, str] = {}
    remaining = set(corpus_ids) if corpus_ids is not None else None
    for offset in range(0, 16316, 100):
        url = f"{base}?{tmpl.format(offset=offset)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
                break
            except Exception as e:  # noqa: BLE001
                _log(f"  url fetch retry offset={offset} attempt={attempt}: {e}")
                time.sleep(1)
        else:
            _log(f"  url fetch GIVING UP on offset {offset}")
            continue
        for r in data["rows"]:
            row = r["row"]
            if remaining is None or row["id"] in remaining:
                urls[row["id"]] = row["video"]["src"]
        if remaining is not None and remaining <= urls.keys():
            break
    return urls


def _mine_chunk(
    chunk: list[str],
    fresh_urls: dict[str, str],
    video_tmp_dir: Path,
    embedding_cache_dir: Path,
    model,
    processor,
) -> int:
    """Refresh-then-immediately-consume one chunk. Returns count recovered."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    def download_one(vid: str) -> tuple[str, bool]:
        path = video_tmp_dir / vid
        if path.exists():
            return vid, True
        if vid not in fresh_urls:
            return vid, False
        try:
            urllib.request.urlretrieve(fresh_urls[vid], path)
            return vid, True
        except Exception:  # noqa: BLE001
            return vid, False

    def decode_one(vid: str):
        try:
            return vid, _sample_frames(str(video_tmp_dir / vid))
        except Exception:  # noqa: BLE001
            return vid, None

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        dl_results = list(ex.map(download_one, chunk))
    ok_ids = [vid for vid, ok in dl_results if ok]

    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as ex:
        dec_results = list(ex.map(decode_one, ok_ids))
    decoded = [(vid, frames) for vid, frames in dec_results if frames is not None]

    recovered = 0
    for start in range(0, len(decoded), MINING_BATCH_SIZE):
        sub = decoded[start : start + MINING_BATCH_SIZE]
        vids = [vid for vid, _ in sub]
        frame_sets = [frames for _, frames in sub]
        inputs = processor.image_processor(frame_sets, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")
        with torch.no_grad():
            out = model.get_video_features(pixel_values=pixel_values)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = torch.nn.functional.normalize(feat, dim=-1).cpu().numpy()
        for vid, vec in zip(vids, feat):
            np.save(embedding_cache_dir / f"{vid}.npy", vec.astype(np.float32))
            recovered += 1

    for vid in chunk:
        p = video_tmp_dir / vid
        if p.exists():
            p.unlink()

    return recovered


def _encode_all_videos(
    needed_videos: list[str],
    embedding_cache_dir: Path,
) -> None:
    """Encode every video in `needed_videos` exactly once, resuming from
    whatever is already cached in `embedding_cache_dir`. Chunked: URLs are
    refreshed for a chunk immediately before that chunk is downloaded, and
    embeddings are checkpointed to disk after every chunk -- safe to
    interrupt and rerun."""
    from transformers import XCLIPModel, XCLIPProcessor

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    video_tmp_dir = embedding_cache_dir / "_tmp"
    video_tmp_dir.mkdir(parents=True, exist_ok=True)

    def emb_path(vid: str) -> Path:
        return embedding_cache_dir / f"{vid}.npy"

    model = None
    processor = None

    for round_no in range(1, MINING_MAX_RETRY_ROUNDS + 1):
        missing = [v for v in needed_videos if not emb_path(v).exists()]
        _log(
            f"mining round {round_no}/{MINING_MAX_RETRY_ROUNDS}: "
            f"{len(needed_videos) - len(missing)} cached, {len(missing)} missing"
        )
        if not missing:
            break

        if model is None:
            _log(f"loading {XCLIP_MODEL_NAME}...")
            model = XCLIPModel.from_pretrained(XCLIP_MODEL_NAME).to("cuda").eval()
            processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_NAME)

        for start in range(0, len(missing), MINING_CHUNK_SIZE):
            chunk = missing[start : start + MINING_CHUNK_SIZE]
            fresh_urls = _fetch_video_urls(set(chunk))
            recovered = _mine_chunk(
                chunk, fresh_urls, video_tmp_dir, embedding_cache_dir, model, processor
            )
            _log(
                f"  chunk {start}-{start + len(chunk)}/{len(missing)}: "
                f"{recovered}/{len(chunk)} recovered this round"
            )

    still_missing = [v for v in needed_videos if not emb_path(v).exists()]
    if still_missing:
        raise RuntimeError(
            f"{len(still_missing)} videos still have no embedding after "
            f"{MINING_MAX_RETRY_ROUNDS} rounds: {still_missing[:20]}"
            f"{'...' if len(still_missing) > 20 else ''}"
        )


def mine_hard_negatives(
    query_ids: list[str],
    qrels: dict[str, str],
    top_ranked: dict[str, list[str]],
    embedding_cache_dir: str | Path,
) -> dict[str, list[str]]:
    """Reproduce the hard-negative mining exactly (see module docstring).

    Resumable and deterministic: `embedding_cache_dir` holds one `.npy` per
    video id; already-cached ids are never re-encoded or re-downloaded, so
    interrupting and rerunning this function picks up where it left off.
    The final ranking is a pure function of the cached embeddings (cosine
    similarity + descending sort), so re-running mining against a complete
    cache always produces the same output.

    Expensive on first run: encodes every unique video referenced by
    `top_ranked` with XCLIP (video-only; never touches source video or
    instruction text). Not called by the default CLI flow -- run
    explicitly with `--remine`.
    """
    import numpy as np

    embedding_cache_dir = Path(embedding_cache_dir)
    needed_videos = sorted(unique_corpus(top_ranked))
    _encode_all_videos(needed_videos, embedding_cache_dir)

    embeddings: dict[str, Any] = {
        vid: np.load(embedding_cache_dir / f"{vid}.npy") for vid in needed_videos
    }

    reduced: dict[str, list[str]] = {}
    for qid in query_ids:
        positive = qrels[qid]
        gallery = top_ranked[qid]
        negatives = [cid for cid in gallery if cid != positive]
        neg_embs = np.stack([embeddings[cid] for cid in negatives])
        sims = neg_embs @ embeddings[positive]
        order = (-sims).argsort(kind="stable")  # stable sort -> deterministic ties
        top_negatives = [negatives[i] for i in order[: REDUCED_GALLERY_SIZE - 1]]
        reduced[qid] = [positive] + top_negatives

    return reduced


# --------------------------------------------------------------------------
# Structural validation (run for both variants before building datasets)
# --------------------------------------------------------------------------


def all_sampled_ids_set(sample: dict[str, list[str]]) -> set[str]:
    return {qid for ids in sample.values() for qid in ids}


def validate(
    query_ids: list[str],
    sample: dict[str, list[str]],
    qrels: dict[str, str],
    top_ranked: dict[str, list[str]],
    expected_gallery_size: int,
    original_top_ranked: dict[str, list[str]] | None = None,
) -> None:
    """All structural checks. Raises AssertionError on any violation."""
    assert len(query_ids) == NUM_QUERIES, (
        f"expected {NUM_QUERIES} queries, got {len(query_ids)}"
    )
    assert set(query_ids) == all_sampled_ids_set(sample), (
        "query_ids do not match the deterministic sample"
    )
    for cat, expected_k in CATEGORY_TARGETS.items():
        actual_k = len(sample.get(cat, []))
        assert actual_k == expected_k, (
            f"category {cat!r}: expected {expected_k}, got {actual_k}"
        )

    assert set(qrels.keys()) == set(query_ids), "qrels missing/extra queries"
    assert set(top_ranked.keys()) == set(query_ids), "top_ranked missing/extra queries"

    for qid in query_ids:
        gallery = top_ranked[qid]
        assert len(gallery) == expected_gallery_size, (
            f"{qid}: expected {expected_gallery_size} candidates, got {len(gallery)}"
        )
        assert len(set(gallery)) == len(gallery), f"{qid}: duplicate candidates"
        positive = qrels[qid]
        assert positive in gallery, f"{qid}: positive {positive!r} not in gallery"
        negatives = [c for c in gallery if c != positive]
        assert len(negatives) == expected_gallery_size - 1, (
            f"{qid}: expected {expected_gallery_size - 1} negatives, got {len(negatives)}"
        )

        if original_top_ranked is not None:
            orig = set(original_top_ranked[qid])
            outside = [c for c in gallery if c not in orig]
            assert not outside, (
                f"{qid}: candidates outside the original 2000-gallery: {outside}"
            )


# --------------------------------------------------------------------------
# Dataset assembly -- targeted per-id downloads (NOT load_dataset() on the
# full corpus/queries configs, which materializes ~52GB of data neither
# variant needs in full and easily exhausts disk on a modest instance).
#
# Fetches signed asset URLs via the lightweight datasets-server /rows API
# (metadata-only, cheap even scanning the full ~16.3k-row corpus) and
# downloads only the specific video files this variant's corpus/queries
# actually reference. Resumable: checkpointed by file existence, chunked
# refresh-then-consume (signed URLs expire, ~1hr TTL -- refreshing a whole
# id set up front and consuming it slowly is exactly what caused mass 403s
# during mining; refreshing per chunk right before download avoids that),
# bounded retries.
# --------------------------------------------------------------------------

ASSET_CHUNK_SIZE = 384
ASSET_MAX_RETRY_ROUNDS = 3


def _fetch_rows_for_ids(
    config: str, split: str, total_rows: int, wanted_ids: set[str]
) -> dict[str, dict]:
    """Fetch full row dicts (all scalar fields + signed asset URLs) for
    `wanted_ids` from a config, via the lightweight datasets-server /rows
    API. Metadata-only cost regardless of video size -- video/audio come
    back as signed URLs, not bytes."""
    import urllib.request

    base = "https://datasets-server.huggingface.co/rows"
    tmpl = (
        f"dataset=mteb%2FOmniCVR&config={config}&split={split}"
        "&offset={offset}&length=100"
    )
    found: dict[str, dict] = {}
    remaining = set(wanted_ids)
    for offset in range(0, total_rows, 100):
        if not remaining:
            break
        url = f"{base}?{tmpl.format(offset=offset)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
                break
            except Exception as e:  # noqa: BLE001
                _log(f"  rows fetch retry offset={offset} attempt={attempt}: {e}")
                time.sleep(1)
        else:
            _log(f"  rows fetch GIVING UP on offset {offset}")
            continue
        for r in data["rows"]:
            row = r["row"]
            if row["id"] in remaining:
                found[row["id"]] = row
                remaining.discard(row["id"])
    return found


def _fetch_corpus_video_urls(corpus_ids: set[str]) -> dict[str, str]:
    rows = _fetch_rows_for_ids("corpus", "corpus", 16316, corpus_ids)
    return {vid: row["video"]["src"] for vid, row in rows.items()}


def _fetch_query_rows(query_ids: set[str]) -> dict[str, dict]:
    """id -> {text, source_id, category, video_url} for the given queries."""
    rows = _fetch_rows_for_ids("queries", "queries", 5000, query_ids)
    return {
        qid: {
            "text": row["text"],
            "source_id": row["source_id"],
            "category": row["category"],
            "video_url": row["video"]["src"],
        }
        for qid, row in rows.items()
    }


def _asset_filename(vid: str, ext: str) -> str:
    """Some ids already carry an extension (corpus ids are like
    'omnicvr_video1.mp4'), others don't (query ids are like 'query-00002').
    Avoid double-extensioning either way."""
    return vid if vid.endswith(f".{ext}") else f"{vid}.{ext}"


def _download_one_asset(vid: str, url: str, dest: Path) -> bool:
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.stat().st_size > 0
    except Exception:  # noqa: BLE001
        if dest.exists():
            dest.unlink()
        return False


def download_assets_resumable(
    ids: list[str],
    url_fetcher,
    asset_dir: Path,
    ext: str = "mp4",
    chunk_size: int = ASSET_CHUNK_SIZE,
    max_retry_rounds: int = ASSET_MAX_RETRY_ROUNDS,
) -> None:
    """Ensure `asset_dir/{id}.{ext}` exists for every id in `ids`.

    Resumable: ids whose file already exists (nonzero size) are skipped
    without any network call. Chunked: `url_fetcher(chunk_ids)` is called
    right before downloading that chunk, so signed URLs are always fresh
    when used. Bounded retries: up to `max_retry_rounds` full passes over
    whatever is still missing, recomputed from actual disk state each round.
    Raises RuntimeError listing anything still missing after all rounds.
    """
    from concurrent.futures import ThreadPoolExecutor

    asset_dir.mkdir(parents=True, exist_ok=True)

    def path_for(vid: str) -> Path:
        return asset_dir / _asset_filename(vid, ext)

    for round_no in range(1, max_retry_rounds + 1):
        missing = [
            v
            for v in ids
            if not (path_for(v).exists() and path_for(v).stat().st_size > 0)
        ]
        _log(
            f"asset download round {round_no}/{max_retry_rounds} "
            f"({asset_dir.name}): {len(ids) - len(missing)} cached, {len(missing)} missing"
        )
        if not missing:
            return

        for start in range(0, len(missing), chunk_size):
            chunk = missing[start : start + chunk_size]
            fresh_urls = url_fetcher(set(chunk))
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
                results = list(
                    ex.map(
                        lambda vid: _download_one_asset(
                            vid, fresh_urls.get(vid, ""), path_for(vid)
                        ),
                        chunk,
                    )
                )
            recovered = sum(1 for r in results if r)
            _log(
                f"  chunk {start}-{start + len(chunk)}/{len(missing)}: "
                f"{recovered}/{len(chunk)} downloaded this chunk"
            )

    still_missing = [
        v for v in ids if not (path_for(v).exists() and path_for(v).stat().st_size > 0)
    ]
    if still_missing:
        raise RuntimeError(
            f"{len(still_missing)} assets still missing after {max_retry_rounds} "
            f"rounds in {asset_dir}: {still_missing[:20]}"
            f"{'...' if len(still_missing) > 20 else ''}"
        )


def build_datasets_targeted(
    query_ids: list[str],
    qrels: dict[str, str],
    top_ranked: dict[str, list[str]],
    asset_cache_dir: str | Path,
) -> dict[str, Dataset]:
    """Assemble corpus/queries/qrels/top_ranked Datasets via targeted
    per-id downloads -- never calls load_dataset() on the full corpus or
    queries configs. Video-only (no audio): the task's declared modalities
    are ["video", "text"]; skipping audio roughly halves the download for
    no evaluation benefit. This is a deliberate simplification versus the
    full base OmniCVR dataset's schema (which also carries an audio
    column) -- flagged here explicitly, not silent.

    `asset_cache_dir` is a persistent, resumable download cache shared
    across variants: since the hard-negatives corpus (~13.7k videos) is a
    strict subset of the mini corpus (~14.4k videos), building mini first
    and then hard-negatives against the same `asset_cache_dir` downloads
    every corpus video at most once.
    """
    asset_cache_dir = Path(asset_cache_dir)

    corpus_ids = sorted(unique_corpus(top_ranked))
    corpus_video_dir = asset_cache_dir / "corpus_video"
    download_assets_resumable(corpus_ids, _fetch_corpus_video_urls, corpus_video_dir)
    corpus = Dataset.from_list(
        [
            {"id": cid, "video": str(corpus_video_dir / _asset_filename(cid, "mp4"))}
            for cid in corpus_ids
        ]
    ).cast_column("video", Video())

    query_rows = _fetch_query_rows(set(query_ids))
    missing_query_meta = set(query_ids) - set(query_rows)
    if missing_query_meta:
        raise RuntimeError(f"no metadata returned for queries: {missing_query_meta}")

    query_video_dir = asset_cache_dir / "query_video"
    download_assets_resumable(
        query_ids,
        lambda chunk: {vid: query_rows[vid]["video_url"] for vid in chunk},
        query_video_dir,
    )
    queries = Dataset.from_list(
        [
            {
                "id": qid,
                "text": query_rows[qid]["text"],
                "video": str(query_video_dir / _asset_filename(qid, "mp4")),
            }
            for qid in query_ids
        ]
    ).cast_column("video", Video())

    qrels_ds = Dataset.from_list(
        [{"query-id": qid, "corpus-id": qrels[qid], "score": 1} for qid in query_ids]
    )

    top_ranked_ds = Dataset.from_list(
        [{"query-id": qid, "corpus-ids": top_ranked[qid]} for qid in query_ids]
    ).cast_column("corpus-ids", Sequence(Value("string")))

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels_ds,
        "top_ranked": top_ranked_ds,
    }


def derive_reduced_corpus(
    mini_corpus_dir: str | Path, reduced_ids: set[str]
) -> Dataset:
    """Build the hard-negatives corpus by filtering the already-saved mini
    corpus dataset (`Dataset.load_from_disk` + `.select()`) rather than
    re-downloading -- the hard-negatives corpus is always a strict subset
    of the mini corpus. Avoids a second full asset download and lets the
    raw video asset cache be freed after mini is saved."""
    from datasets import load_from_disk

    mini_corpus = load_from_disk(str(mini_corpus_dir))
    idx = {cid: i for i, cid in enumerate(mini_corpus["id"])}
    missing = reduced_ids - set(idx)
    if missing:
        raise RuntimeError(
            f"{len(missing)} reduced-corpus ids not found in mini corpus: "
            f"{sorted(missing)[:20]}"
        )
    return mini_corpus.select([idx[cid] for cid in sorted(reduced_ids)])


def push_datasets(
    datasets: dict[str, Dataset], repo_id: str, num_proc: int | None = None
) -> None:
    for name, ds in datasets.items():
        DatasetDict({"test": ds}).push_to_hub(repo_id, name, num_proc=num_proc)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_DEFAULT_REPO_IDS = {
    "mini": "mteb/OmniCVR-mini",
    "mini-hard-negatives": "mteb/OmniCVR-mini-hard-negatives",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["mini", "mini-hard-negatives"], required=True
    )
    parser.add_argument("--repo-id", default=None, help="Defaults per --variant.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run sampling + structural validation without loading video data or building datasets.",
    )
    parser.add_argument(
        "--mined-top-ranked-path",
        default=None,
        help="For --variant mini-hard-negatives: path to a precomputed mining "
        "result (query-id -> list[corpus-id], positive first). Not checked "
        "into this repo. Required unless --remine is passed.",
    )
    parser.add_argument(
        "--remine",
        action="store_true",
        help="Recompute hard negatives with XCLIP instead of loading "
        "--mined-top-ranked-path. Expensive (encodes ~14.4k videos on GPU) "
        "but resumable -- safe to interrupt and rerun.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default="./omnicvr_xclip_embeddings",
        help="Where --remine caches per-video XCLIP embeddings (resumable).",
    )
    parser.add_argument(
        "--asset-cache-dir",
        default="./omnicvr_asset_cache",
        help="Where targeted per-id video downloads are cached (resumable). "
        "Shared across variants -- build mini first, then mini-hard-negatives "
        "against the same dir, and the corpus videos already downloaded for "
        "mini (a superset) are reused with no re-download.",
    )
    parser.add_argument(
        "--reduce-corpus-from",
        default=None,
        help="For --variant mini-hard-negatives: path to an already-saved "
        "mini corpus dataset (via --save-to-disk-dir on a prior `--variant "
        "mini` run). If given, the hard-negatives corpus is derived by "
        "filtering that saved dataset instead of downloading video again.",
    )
    parser.add_argument(
        "--save-to-disk-dir",
        default=None,
        help="If given, save each of corpus/queries/qrels/top_ranked to "
        "<dir>/<config> via Dataset.save_to_disk (local, HF-ready form, no "
        "push).",
    )
    parser.add_argument("--num-proc", type=int, default=None)
    args = parser.parse_args()

    if (
        args.variant == "mini-hard-negatives"
        and not args.remine
        and not args.mined_top_ranked_path
    ):
        parser.error(
            "--variant mini-hard-negatives requires either --mined-top-ranked-path "
            "(use an already-mined result) or --remine (recompute it)."
        )

    id_to_category = load_query_categories()
    sample = sample_500_queries(id_to_category)
    query_ids = all_sampled_ids(sample)
    print(
        f"sampled {len(query_ids)} queries: "
        f"{ {cat: len(ids) for cat, ids in sample.items()} }"
    )

    qrels = load_qrels_for(set(query_ids), args.num_proc)
    original_top_ranked = load_top_ranked_for(set(query_ids), args.num_proc)

    if args.variant == "mini":
        top_ranked = original_top_ranked
        expected_gallery_size = FULL_GALLERY_SIZE
        orig_for_validation = None
    else:
        if args.remine:
            top_ranked = mine_hard_negatives(
                query_ids, qrels, original_top_ranked, args.embedding_cache_dir
            )
        else:
            top_ranked = load_mined_top_ranked(args.mined_top_ranked_path)
        expected_gallery_size = REDUCED_GALLERY_SIZE
        orig_for_validation = original_top_ranked

    validate(
        query_ids,
        sample,
        qrels,
        top_ranked,
        expected_gallery_size,
        original_top_ranked=orig_for_validation,
    )
    print("all structural validations passed")
    print(f"unique corpus videos: {len(unique_corpus(top_ranked))}")

    if args.validate_only:
        return

    datasets = build_datasets_targeted(
        query_ids, qrels, top_ranked, args.asset_cache_dir
    )

    if args.variant == "mini-hard-negatives" and args.reduce_corpus_from:
        reduced_ids = unique_corpus(top_ranked)
        datasets["corpus"] = derive_reduced_corpus(args.reduce_corpus_from, reduced_ids)
        print(f"derived corpus from {args.reduce_corpus_from} (no re-download)")

    for name, ds in datasets.items():
        print(f"{name}: {len(ds)} rows, features={ds.features}")

    if args.save_to_disk_dir:
        out_dir = Path(args.save_to_disk_dir)
        for name, ds in datasets.items():
            path = out_dir / name
            print(f"saving {name} ({len(ds)} rows) to {path}")
            ds.save_to_disk(str(path))

    if args.push_to_hub:
        repo_id = args.repo_id or _DEFAULT_REPO_IDS[args.variant]
        push_datasets(datasets, repo_id, args.num_proc)
        print(f"pushed to {repo_id}")
    else:
        print("skipped push (pass --push-to-hub to upload)")


if __name__ == "__main__":
    main()
