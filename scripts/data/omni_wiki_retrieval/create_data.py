"""Build the OmniWiki video-to-page retrieval dataset (OmniWikiV2IRetrieval / OmniWikiV2TRetrieval).

The query is a video embedded on a Wikipedia page; the gold doc is that page -- a screenshot (V2I)
or its text (V2T). Both tasks share one Hub dataset (whybe-choi/OmniWikiRetrieval) whose corpus
carries {id, image, text}; each task keeps its column via dataset_transform.

Sources (row-aligned; doc_id == 0-based row index):
  wiki-ss-corpus      title + text (text = the V2T doc, stored as "{title} {text}")
  wiki-ss-corpus-new  image + docid (== the row index wiki-ss-nq-new's document ids reference)
  wiki-ss-nq-new      train queries with positive_document_ids / negative_document_ids

Query videos live on pages that are wiki-ss-nq-new train positives; corpus = those gold pages plus
their BM25 hard negatives. A video is kept only if 10-180s AND decodable by torchcodec (202
undecodable Theora .ogv are dropped; see UNDECODABLE_CACHE), and qrels are complete over the
corpus. Reference counts (2026-09): 1,087 queries / 40,861 corpus / 1,191 qrels / 980 relevant
pages.

`python create_data.py` writes the mapping files (queries/corpus_docids/corpus_titles/qrels/stats).
--materialize / --push-to-hub are a streaming helper for the media upload; the shipped dataset was
pushed via MTEB's push_dataset_to_hub. Set a real contact URL in UA (Wikimedia 403s otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator

import requests
from datasets import Dataset, Features, Image, Value, load_dataset

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
CORPUS_TITLE = "Tevatron/wiki-ss-corpus"  # title/text
CORPUS_IMAGE = "Tevatron/wiki-ss-corpus-new"  # image + sequential docid (== row index)
QUERIES = "Tevatron/wiki-ss-nq-new"  # train positive/negative document ids

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia UA policy is enforced on upload.wikimedia.org (403 without a contact URL); use your own.
UA = "OmniWikiRetrievalBuilder/1.0 (https://github.com/embeddings-benchmark/mteb; mteb dataset build)"
# .ogg is intentionally excluded (overwhelmingly audio, indistinguishable from video via prop=images).
VIDEO_EXT = (".webm", ".ogv")
API_BATCH = 50  # titles per prop=images request
# Duration filter (not a clip): keep a video only if VIDEO_MIN_SEC <= dur <= VIDEO_MAX_SEC
# (unknown-duration excluded); kept videos are used in full.
VIDEO_MIN_SEC = 10
VIDEO_MAX_SEC = 180
MAX_VIDEOS_PER_PAGE = 30  # cap list/series pages to <=30 videos/page
SAMPLE_SEED = 42  # seed for random per-page video sampling

# Reference counts (2026-09 sweep, duration filter 10-180s + torchcodec-decodability filter);
# a mismatch is reported (not fatal) so drift in the live sweep is visible.
EXPECTED = {"queries": 1087, "corpus": 40861, "qrels_rows": 1191, "relevant_pages": 980}

# Paths are anchored to this file so re-runs from anywhere hit the same cache/output.
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("OMNI_WIKI_CACHE", os.path.join(_HERE, "cache"))
OUT_DIR = os.environ.get("OMNI_WIKI_OUT", os.path.join(_HERE, "output"))
os.makedirs(CACHE, exist_ok=True)

# File titles torchcodec (MTEB's Video() backend) cannot decode, dropped like the duration filter.
# Produced during materialization; caching it keeps the build reproducible without re-downloading.
# Absent cache -> decodability filter skipped (a warning is printed).
UNDECODABLE_CACHE = os.path.join(CACHE, "undecodable_videos.json")


def _read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: never leave a half-written cache


# --------------------------------------------------------------------------------------
# Step 1. Corpus titles, indexed by row position (== doc_id)
# --------------------------------------------------------------------------------------
def load_titles() -> list[str]:
    """titles[i] is the title of document i (== wiki-ss-corpus-new docid i).

    Streams the corpus and keeps only ``title``: the repo is ~394 GB of screenshots and we
    must not download it just to read a list of strings. Row order is the shard order
    (data-00000..00808), which is the order wiki-ss-nq-new's document ids index into.
    """
    path = os.path.join(CACHE, "titles.json")
    cached = _read_json(path)
    if cached is not None:
        return cached
    ds = load_dataset(CORPUS_TITLE, split="train", streaming=True)
    ds = ds.select_columns(["title"]) if hasattr(ds, "select_columns") else ds
    titles = [row["title"] for row in ds]
    _write_json(path, titles)
    return titles


# --------------------------------------------------------------------------------------
# Step 2. Detect video-bearing pages via the Wikipedia API and map video -> page(s)
# --------------------------------------------------------------------------------------
def wiki_get(params: dict) -> dict | None:
    """One API call with maxlag/network retries. Returns None only after all retries fail
    or when the API answers with a non-maxlag error (caller must treat that as a failure)."""
    params = {**params, "format": "json", "maxlag": "5"}
    for attempt in range(6):
        try:
            # POST: 50 titles can exceed safe GET URL lengths.
            r = requests.post(
                WIKI_API, data=params, headers={"User-Agent": UA}, timeout=60
            )
            data = r.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
            continue
        err = data.get("error") if isinstance(data, dict) else None
        if err is not None:
            if err.get("code") in {"maxlag", "ratelimited"}:
                time.sleep(2 * (attempt + 1))
                continue
            return None  # hard error: do not silently swallow
        return data
    return None


def _resolved_to_queried(
    query_block: dict, batch: list[str], acc: dict[str, set[str]]
) -> None:
    """Accumulate resolved-title -> {queried corpus titles} using the API's `normalized`
    and `redirects` arrays (redirects=1 returns pages under the *target* title)."""
    for t in batch:
        acc.setdefault(t, set()).add(t)
    for key in ("normalized", "redirects"):
        for m in query_block.get(key, []) or []:
            src, dst = m.get("from"), m.get("to")
            if src is None or dst is None:
                continue
            acc.setdefault(dst, set()).update(acc.get(src, {src}))


def _sweep_batch(batch: list[str]) -> dict[str, list[str]] | None:
    """{corpus_title: [video files]} for one batch, or None if any request failed."""
    videos_by_resolved: dict[str, set[str]] = defaultdict(set)
    rev: dict[str, set[str]] = {}
    cont: dict = {}
    while True:
        data = wiki_get(
            {
                "action": "query",
                "prop": "images",
                "imlimit": "max",
                "redirects": "1",
                "titles": "|".join(batch),
                **cont,
            }
        )
        if data is None or "query" not in data:
            return None
        q = data["query"] or {}
        _resolved_to_queried(q, batch, rev)
        for page in (q.get("pages") or {}).values():
            if "missing" in page or "invalid" in page:
                continue
            for im in page.get("images", []) or []:
                fn = im.get("title", "")
                if fn.lower().endswith(VIDEO_EXT):
                    videos_by_resolved[page["title"]].add(fn)
        if "continue" in data:
            cont = data["continue"]
        else:
            break

    batch_set = set(batch)
    out: dict[str, set[str]] = defaultdict(set)
    for resolved, files in videos_by_resolved.items():
        # Key by the corpus title(s) that led here, so later title lookups match the corpus.
        for queried in rev.get(resolved, {resolved}) & batch_set:
            out[queried].update(files)
    return {t: sorted(v) for t, v in out.items()}


def sweep_videos(titles: list[str]) -> dict[str, list[str]]:
    """Return {corpus_title: [video_file, ...]} for pages that carry >=1 .webm/.ogv file.

    Checkpointed per batch in ``sweep_batches.jsonl`` (resumable). Failed batches are
    recorded and retried on the next run; the final ``page_videos.json`` is written only once
    every batch has succeeded, so an incomplete sweep can never masquerade as complete.
    """
    final = os.path.join(CACHE, "page_videos.json")
    done = _read_json(final)
    if done is not None:
        return done

    uniq = list(dict.fromkeys(t for t in titles if t))
    n_batches = (len(uniq) + API_BATCH - 1) // API_BATCH
    ckpt = os.path.join(CACHE, "sweep_batches.jsonl")

    results: dict[int, dict[str, list[str]]] = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                results[rec["batch"]] = rec["pages"]
    print(f"  sweep: {len(results)}/{n_batches} batches already done")

    failed: list[int] = []
    with open(ckpt, "a", encoding="utf-8") as f:
        for b in range(n_batches):
            if b in results:
                continue
            batch = uniq[b * API_BATCH : (b + 1) * API_BATCH]
            res = _sweep_batch(batch)
            if res is None:
                failed.append(b)
                continue
            results[b] = res
            f.write(json.dumps({"batch": b, "pages": res}, ensure_ascii=False) + "\n")
            f.flush()
            if b % 200 == 0:
                print(f"  swept batch {b}/{n_batches} ({b * API_BATCH} titles)")

    _write_json(os.path.join(CACHE, "sweep_failed.json"), failed)
    if failed:
        raise RuntimeError(
            f"sweep incomplete: {len(failed)} batches failed (see cache/sweep_failed.json); "
            "re-run to retry them. page_videos.json was NOT written."
        )

    page_videos: dict[str, set[str]] = defaultdict(set)
    for res in results.values():
        for t, v in res.items():
            page_videos[t].update(v)
    out = {p: sorted(v) for p, v in page_videos.items()}
    _write_json(final, out)
    return out


# --------------------------------------------------------------------------------------
# Step 3. wiki-ss-nq-new train: per-query positive / negative document ids (row indices)
# --------------------------------------------------------------------------------------
def load_train_queries():
    # each row: query_id, query_text, answers, positive_document_ids, negative_document_ids
    return load_dataset(QUERIES, split="train")


def _imageinfo(files: list[str], iiprop: str) -> Iterator[tuple[str, dict]]:
    """Yield (queried File title, imageinfo dict) for ``files`` via batched prop=imageinfo,
    mapping the response's (possibly normalized/redirected) title back to the queried one.
    Batches whose request failed are skipped, so callers see them as missing."""
    for start in range(0, len(files), API_BATCH):
        batch = files[start : start + API_BATCH]
        data = wiki_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": iiprop,
            }
        )
        if not data or "query" not in data:
            continue
        q = data["query"] or {}
        rev: dict[str, set[str]] = {}
        _resolved_to_queried(q, batch, rev)
        for page in (q.get("pages") or {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            for queried in rev.get(page["title"], {page["title"]}):
                yield queried, ii


def video_durations(video_files: Iterable[str]) -> dict[str, float]:
    """Commons duration (seconds) per video File title, via imageinfo (iiprop=size).
    Cached in ``durations.json``; entries missing after a failed batch are retried next run."""
    path = os.path.join(CACHE, "durations.json")
    cached: dict[str, float] = _read_json(path, {})
    todo = [f for f in video_files if f not in cached]
    for queried, ii in _imageinfo(todo, "size|mediatype"):
        if ii.get("mediatype") == "VIDEO" and isinstance(
            ii.get("duration"), (int, float)
        ):
            cached[queried] = ii["duration"]
    _write_json(path, cached)
    return cached


# --------------------------------------------------------------------------------------
# Step 4-6. Assemble queries / corpus / qrels under the "train-positive video" scope
# --------------------------------------------------------------------------------------
def build(titles, page_videos, train):
    """Everything here is in row-index space; doc_id = str(row index).

    Query videos are filtered by duration to [VIDEO_MIN_SEC, VIDEO_MAX_SEC]; a gold page
    survives only if it still carries >=1 in-range video, and the corpus/hard-negatives
    cascade from the surviving gold set.
    """
    # 4a) candidate gold rows = train-positive rows whose title carries any video; remember the
    #     relevant train queries (positive hits a video page) so we can pull their hard negs later.
    cand_gold_idx: set[int] = set()
    relevant_rows: list[tuple[list[int], list[int]]] = []
    for row in train:
        pos = [int(d) for d in row["positive_document_ids"]]
        hit = [i for i in pos if titles[i] in page_videos]
        if not hit:
            continue
        cand_gold_idx.update(hit)
        relevant_rows.append((pos, [int(d) for d in row["negative_document_ids"]]))

    # 4b) duration + decodability filter. Fetch durations for every video on a candidate gold
    #     page, then keep only videos that are in-range AND decodable by torchcodec; a candidate
    #     gold row survives iff it has >=1 such usable video.
    cand_videos = sorted({v for i in cand_gold_idx for v in page_videos[titles[i]]})
    durations = video_durations(cand_videos)
    undecodable = set(_read_json(UNDECODABLE_CACHE, []) or [])
    if not undecodable:
        print(
            "  NOTE: no undecodable-video cache found; skipping decodability filter "
            f"(expected at {UNDECODABLE_CACHE})"
        )

    def in_range(v: str) -> bool:
        d = durations.get(v)
        return isinstance(d, (int, float)) and VIDEO_MIN_SEC <= d <= VIDEO_MAX_SEC

    def usable(v: str) -> bool:
        return in_range(v) and v not in undecodable

    gold_idx: set[int] = {
        i for i in cand_gold_idx if any(usable(v) for v in page_videos[titles[i]])
    }

    # 4c) hard negatives = negatives of the relevant queries that still hit a surviving gold row.
    hardneg_idx: set[int] = set()
    for pos, neg in relevant_rows:
        if any(i in gold_idx for i in pos):
            hardneg_idx.update(neg)

    # 6) corpus = gold rows UNION hard-negative rows (as row indices / doc_ids)
    corpus_idx: set[int] = gold_idx | hardneg_idx
    corpus = {str(i): titles[i] for i in sorted(corpus_idx)}

    # 5a) QUERY SET: per gold page, the in-range videos, capped to <=MAX_VIDEOS_PER_PAGE
    #     (reproducible per-page random sample; duplicate-title rows draw the identical sample).
    query_videos: set[str] = set()
    for i in sorted(gold_idx):
        page = titles[i]
        vfiles = sorted(v for v in page_videos[page] if usable(v))
        if MAX_VIDEOS_PER_PAGE and len(vfiles) > MAX_VIDEOS_PER_PAGE:
            rng = random.Random(f"{SAMPLE_SEED}:{page}")
            vfiles = rng.sample(vfiles, MAX_VIDEOS_PER_PAGE)
        query_videos.update(vfiles)

    # 5b) QRELS, complete over the corpus: any corpus row that embeds a query video is
    #     relevant to it -- including hard-negative rows and gold rows whose copy of the video
    #     was capped out. A video that survives as a query never loses a label.
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for i in sorted(corpus_idx):
        for v in page_videos.get(titles[i], ()):
            if v in query_videos:
                qrels[v][str(i)] = 1
    assert set(qrels) == query_videos, "every query must have >=1 relevant corpus doc"

    # 5c) query metadata. Every kept video is in [VIDEO_MIN_SEC, VIDEO_MAX_SEC] and used in full
    #     (no trimming); duration is recorded for reference.
    queries: dict[str, dict] = {
        v: {"file": v, "duration": durations.get(v)} for v in sorted(query_videos)
    }

    relevant_idx = {int(d) for docs in qrels.values() for d in docs}
    stats = {
        "queries": len(queries),
        "corpus": len(corpus),
        "qrels_rows": sum(len(d) for d in qrels.values()),
        "relevant_pages": len(relevant_idx),
        "gold_rows": len(gold_idx),
        "hardneg_relabeled_relevant": len(relevant_idx - gold_idx),
        "multi_gold_queries": sum(len(d) > 1 for d in qrels.values()),
    }
    return queries, corpus, dict(qrels), stats


# --------------------------------------------------------------------------------------
# Step 7. Emit files (image/video materialization is a downstream step; see module docstring)
# --------------------------------------------------------------------------------------
def _qrels_rows(qrels) -> Iterator[dict]:
    """MTEB loader shape for the qrels config: one row per (query-id, corpus-id, score)."""
    for qid in sorted(qrels):
        for did, score in sorted(qrels[qid].items(), key=lambda kv: int(kv[0])):
            yield {"query-id": qid, "corpus-id": did, "score": score}


def emit(queries, corpus, qrels, stats, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    _write_json(
        os.path.join(out_dir, "corpus_docids.json"),
        {"num_docs": len(corpus), "doc_ids": sorted(corpus, key=int)},
    )
    _write_json(os.path.join(out_dir, "corpus_titles.json"), corpus)
    _write_json(os.path.join(out_dir, "queries.json"), queries)
    _write_json(os.path.join(out_dir, "qrels.json"), qrels)
    with open(os.path.join(out_dir, "qrels_mteb.jsonl"), "w", encoding="utf-8") as f:
        for row in _qrels_rows(qrels):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(os.path.join(out_dir, "stats.json"), stats)

    for k, v in stats.items():
        print(f"{k:28s} = {v:,}")
    drift = {k: (stats[k], v) for k, v in EXPECTED.items() if stats[k] != v}
    if drift:
        print(
            f"WARNING: stats differ from reference (got, expected): {drift}\n"
            "         Wikipedia content drifts over time; the reference numbers assume the\n"
            "         2026-09 sweep cache. Update EXPECTED if the new sweep is intended."
        )


# --------------------------------------------------------------------------------------
# Step 8. Materialize media (corpus images + query videos) and optionally push to the Hub
# --------------------------------------------------------------------------------------
def _safe_name(title: str) -> str:
    return re.sub(r"[^\w.-]+", "_", title)[:150]


def _download(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True  # resume: already fetched
    tmp = dest + ".part"
    for attempt in range(4):
        try:
            with requests.get(
                url, headers={"User-Agent": UA}, stream=True, timeout=180
            ) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def _commons_urls(files: list[str]) -> dict[str, str]:
    """Direct download URL per Commons/Wikipedia File: title (batched)."""
    return {
        queried: ii["url"] for queried, ii in _imageinfo(files, "url") if ii.get("url")
    }


def materialize_corpus_images(corpus_ids):
    """Dataset of {id, image} for the corpus docs. Streams wiki-ss-corpus-new and keeps rows
    whose docid (== row index) is in the corpus. NOTE: streaming transfers the whole corpus-new
    to keep ~len(corpus_ids) rows -- run on a machine with bandwidth; only kept rows are held.
    This is the streaming reference; the shipped ``corpus`` config additionally carries ``text``
    (``"{title} {text}"`` per docid, for V2T)."""
    wanted = set(corpus_ids)
    stream = load_dataset(CORPUS_IMAGE, split="train", streaming=True)

    def gen():
        for row in stream:
            did = str(row["docid"])
            if did in wanted:
                yield {"id": did, "image": row["image"]}

    return Dataset.from_generator(
        gen, features=Features({"id": Value("string"), "image": Image()})
    )


def materialize_query_videos(query_ids, out_dir):
    """Dataset of {id, video} -- download each Commons video (already <= VIDEO_MAX_SEC) to disk.
    Resumable (skips files already present)."""
    vids_dir = os.path.join(out_dir, "videos")
    os.makedirs(vids_dir, exist_ok=True)
    urls = _commons_urls(sorted(query_ids))
    rows, missing = [], []
    for vid in sorted(query_ids):
        url = urls.get(vid)
        if not url:
            missing.append(vid)
            continue
        ext = os.path.splitext(url)[1] or ".webm"
        dest = os.path.join(vids_dir, _safe_name(vid) + ext)
        if _download(url, dest):
            rows.append({"id": vid, "video": dest})
        else:
            missing.append(vid)
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(query_ids)} query videos could not be downloaded"
        )
    ds = Dataset.from_list(rows)
    try:
        from datasets import Video

        ds = ds.cast_column("video", Video())
    except Exception:
        print(
            "NOTE: datasets.Video unavailable; 'video' column holds local file paths."
        )
    return ds


def qrels_dataset(qrels):
    return Dataset.from_list(list(_qrels_rows(qrels)))


def push(repo_id, corpus_ds, queries_ds, qrels_ds, split, private):
    """Streaming-reference upload. The shipped whybe-choi/OmniWikiRetrieval was instead pushed via
    MTEB's ``AbsTaskRetrieval.push_dataset_to_hub`` (configs ``corpus``/``queries``/``qrels``, all
    on split ``test``, corpus carrying {id, image, text})."""
    corpus_ds.push_to_hub(
        repo_id, config_name="corpus", split="corpus", private=private
    )
    queries_ds.push_to_hub(repo_id, config_name="queries", split=split, private=private)
    qrels_ds.push_to_hub(repo_id, config_name="default", split=split, private=private)
    print(f"pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the OmniWiki video-to-page retrieval dataset (V2I / V2T)."
    )
    ap.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="HF dataset repo id to push to (e.g. 'you/OmniWikiRetrieval'). "
        "Implies --materialize. If omitted, only local mapping files are written.",
    )
    ap.add_argument(
        "--materialize",
        action="store_true",
        help="download corpus images + query videos locally (heavy).",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="split name for the queries/qrels configs (default: test; eval-only task).",
    )
    ap.add_argument(
        "--private", action="store_true", help="push the Hub repo as private."
    )
    ap.add_argument(
        "--out-dir", default=OUT_DIR, help="output directory for local files."
    )
    args = ap.parse_args()

    titles = load_titles()
    print(f"corpus rows: {len(titles):,}")
    page_videos = sweep_videos(titles)
    print(f"video-bearing pages: {len(page_videos):,}")
    train = load_train_queries()
    print(f"train queries: {len(train):,}")
    queries, corpus, qrels, stats = build(titles, page_videos, train)
    emit(queries, corpus, qrels, stats, out_dir=args.out_dir)

    if args.materialize or args.push_to_hub:
        print("materializing corpus images (streaming wiki-ss-corpus-new)...")
        corpus_ds = materialize_corpus_images(set(corpus))
        print(f"  corpus images: {len(corpus_ds):,}")
        print("materializing query videos (downloading from Commons)...")
        queries_ds = materialize_query_videos(set(queries), args.out_dir)
        print(f"  query videos: {len(queries_ds):,}")
        qrels_ds = qrels_dataset(qrels)
        if args.push_to_hub:
            push(
                args.push_to_hub,
                corpus_ds,
                queries_ds,
                qrels_ds,
                args.split,
                args.private,
            )
        else:
            corpus_ds.save_to_disk(os.path.join(args.out_dir, "hf_corpus"))
            queries_ds.save_to_disk(os.path.join(args.out_dir, "hf_queries"))
            qrels_ds.save_to_disk(os.path.join(args.out_dir, "hf_qrels"))
            print(
                f"saved HF datasets under {args.out_dir} (pass --push-to-hub to upload)"
            )


if __name__ == "__main__":
    main()
