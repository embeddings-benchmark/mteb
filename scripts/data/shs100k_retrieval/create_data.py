#!/usr/bin/env python3
"""Package classic NovaFrost SHS100K-TEST into MTEB a2a retrieval format.

Source metadata (YouTube URLs only):
  https://github.com/NovaFrost/SHS100K

Joins SHS100K-TEST (set_id, ver_id) with `list`, downloads audio-only via
yt-dlp (resume under --work-dir/audio/), then builds corpus / queries / qrels.
Cliques with fewer than 2 successfully downloaded tracks are dropped.

Usage:
  uv pip install yt-dlp
  export HF_TOKEN=...
  # Resume + speed: parallel workers, no inter-download sleep by default
  uv run python scripts/data/shs100k_retrieval/create_data.py \\
      --repo-id {repo_id}/SHS100K-A2A \\
      --work-dir /tmp/shs100k_mteb \\
      --workers 8 \\
      --push
  # If YouTube rate-limits, drop workers and/or add pacing:
  #   --workers 2 --sleep-interval 1 --max-sleep-interval 3
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import create_repo
from tqdm import tqdm

_LIST_URL = "https://raw.githubusercontent.com/NovaFrost/SHS100K/master/list"
_TEST_URL = "https://raw.githubusercontent.com/NovaFrost/SHS100K/master/SHS100K-TEST"
_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)(?P<id>[\w-]{11})",
    re.IGNORECASE,
)


class SkipTrack(Exception):
    """Non-fatal download skip."""


def _resolve_yt_dlp() -> list[str]:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        binary = shutil.which("yt-dlp")
        if binary:
            probe = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                return [binary]
        raise SystemExit(
            "yt-dlp is not available for this Python interpreter.\n"
            "Install into the project env:\n"
            "  uv pip install yt-dlp\n"
            "Then re-run with: uv run python scripts/data/shs100k_retrieval/create_data.py ..."
        ) from None
    return [sys.executable, "-m", "yt_dlp"]


def _fetch_text(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    dest.write_text(text, encoding="utf-8")
    return text


def _youtube_id(url: str) -> str | None:
    m = _YT_ID_RE.search(url)
    return m.group("id") if m else None


def _audio_is_decodable(path: Path) -> bool:
    """Reject incomplete/corrupt containers (e.g. m4a missing moov atom)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # Fallback: torchcodec (same decoder HF datasets uses).
        try:
            from torchcodec.decoders import AudioDecoder
        except ImportError:
            return path.exists() and path.stat().st_size > 0
        try:
            _ = AudioDecoder(str(path)).metadata
            return True
        except Exception:
            return False
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _load_metadata(
    meta_dir: Path,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict]]:
    test_text = _fetch_text(_TEST_URL, meta_dir / "SHS100K-TEST")
    list_text = _fetch_text(_LIST_URL, meta_dir / "list")

    test_pairs: list[tuple[str, str]] = []
    for line in test_text.splitlines():
        line = line.strip()
        if not line:
            continue
        sid, vid = line.split("\t")[:2]
        test_pairs.append((sid, vid))

    meta: dict[tuple[str, str], dict] = {}
    for line in list_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        sid, vid, title, artist, url = parts[0], parts[1], parts[2], parts[3], parts[4]
        yid = _youtube_id(url)
        if yid is None:
            continue
        meta[(sid, vid)] = {
            "title": title,
            "artist": artist,
            "url": url,
            "youtube_id": yid,
        }
    return test_pairs, meta


def _download_audio(
    url: str,
    out: Path,
    *,
    yt_dlp: list[str],
    cookies_from_browser: str | None,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    concurrent_fragments: int,
) -> None:
    if out.exists() and out.stat().st_size > 0 and _audio_is_decodable(out):
        return
    if out.exists():
        # Incomplete download (e.g. m4a missing moov) — force re-fetch.
        out.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer native audio container; remux to m4a only when needed (avoids slow re-encode).
    out_tmpl = str(out.with_suffix(""))
    last_err = ""
    for attempt in range(retries + 1):
        # Sleep only on retries by default; optional inter-download pacing via sleep_min.
        if attempt:
            delay = random.uniform(sleep_min, max(sleep_min, sleep_max))
            delay *= 2 ** (attempt - 1)
            time.sleep(delay)
        elif sleep_min > 0:
            time.sleep(random.uniform(sleep_min, max(sleep_min, sleep_max)))
        cmd = [
            *yt_dlp,
            "-f",
            "ba/b",
            "-x",
            "--audio-format",
            "m4a",
            "--audio-quality",
            "0",
            "--no-warnings",
            "--no-playlist",
            "--no-mtime",
            "--concurrent-fragments",
            str(max(1, concurrent_fragments)),
            "-o",
            f"{out_tmpl}.%(ext)s",
        ]
        if cookies_from_browser:
            cmd.extend(["--cookies-from-browser", cookies_from_browser])
        cmd.append(url)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # yt-dlp may write .m4a or another ext then convert
        candidates = list(out.parent.glob(f"{out.stem}.*"))
        ok = out.exists() and out.stat().st_size > 0
        if not ok:
            for c in candidates:
                if (
                    c.suffix.lower() in {".m4a", ".mp3", ".opus", ".webm", ".ogg"}
                    and c.stat().st_size > 0
                ):
                    if c != out:
                        c.replace(out)
                    ok = out.exists() and out.stat().st_size > 0
                    break
        if proc.returncode == 0 and ok and _audio_is_decodable(out):
            return
        last_err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        if ok and not _audio_is_decodable(out):
            last_err = (last_err + "; undecodable audio").strip("; ")
        for c in candidates:
            if c != out:
                c.unlink(missing_ok=True)
        if out.exists():
            out.unlink(missing_ok=True)
        low = last_err.lower()
        if any(
            s in low
            for s in (
                "private video",
                "video unavailable",
                "been removed",
                "not available",
                "copyright",
                "status code is 404",
                "sign in to confirm",
                "ip address is blocked",
            )
        ):
            break
    raise SkipTrack(last_err[-500:] if last_err else "yt-dlp failed")


def _download_one(
    sid: str,
    vid: str,
    info: dict,
    audio_dir: Path,
    *,
    yt_dlp: list[str],
    cookies_from_browser: str | None,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    concurrent_fragments: int,
) -> tuple[str, str, str, Path | None, str | None]:
    """Returns (sid, vid, yid, path_or_None, error_or_None)."""
    yid = info["youtube_id"]
    out = audio_dir / f"{yid}.m4a"
    if out.exists() and out.stat().st_size > 0 and _audio_is_decodable(out):
        return sid, vid, yid, out, None
    try:
        _download_audio(
            info["url"],
            out,
            yt_dlp=yt_dlp,
            cookies_from_browser=cookies_from_browser,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            retries=retries,
            concurrent_fragments=concurrent_fragments,
        )
        return sid, vid, yid, out, None
    except SkipTrack as e:
        return sid, vid, yid, None, str(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Wissam42/SHS100K-A2A")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/shs100k_mteb"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel yt-dlp downloads (default 8). Lower if YouTube rate-limits.",
    )
    parser.add_argument(
        "--concurrent-fragments",
        type=int,
        default=4,
        help="yt-dlp concurrent fragment downloads per video (default 4).",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.0,
        help="Min seconds between downloads / before retries (default 0).",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=1.0,
        help="Max seconds for sleep jitter (default 1).",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Pass through to yt-dlp, e.g. chrome / firefox / safari.",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work: Path = args.work_dir.resolve()
    audio_dir = work / "audio"
    meta_dir = work / "meta"
    audio_dir.mkdir(parents=True, exist_ok=True)

    yt_dlp = _resolve_yt_dlp()
    print(f"Using yt-dlp={yt_dlp} workers={args.workers}")

    test_pairs, meta = _load_metadata(meta_dir)
    rows: list[tuple[str, str, dict]] = []
    missing_meta = 0
    for sid, vid in test_pairs:
        info = meta.get((sid, vid))
        if info is None:
            missing_meta += 1
            continue
        rows.append((sid, vid, info))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(
        f"TEST tracks={len(test_pairs)} with_meta={len(rows)} "
        f"missing_meta={missing_meta}"
    )

    # Deduplicate by youtube_id so parallel workers never fetch the same video twice.
    path_by_yid: dict[str, Path] = {}
    ok_by_key: dict[tuple[str, str], Path] = {}
    unique_jobs: list[tuple[str, str, dict]] = []
    alias_jobs: list[tuple[str, str, dict]] = []
    queued_yid: set[str] = set()
    n_resume = 0
    n_fail = 0
    fail_log = work / "download_failures.txt"
    fail_lines: list[str] = []

    n_corrupt = 0
    for sid, vid, info in rows:
        yid = info["youtube_id"]
        out = audio_dir / f"{yid}.m4a"
        if out.exists() and out.stat().st_size > 0 and _audio_is_decodable(out):
            n_resume += 1
            path_by_yid[yid] = out
            ok_by_key[(sid, vid)] = out
            continue
        if out.exists() and out.stat().st_size > 0:
            # Leave file for _download_audio to delete + re-fetch.
            n_corrupt += 1
        if yid in path_by_yid:
            ok_by_key[(sid, vid)] = path_by_yid[yid]
            continue
        if yid in queued_yid:
            alias_jobs.append((sid, vid, info))
        else:
            queued_yid.add(yid)
            unique_jobs.append((sid, vid, info))

    print(
        f"resume={n_resume} corrupt_to_refetch={n_corrupt} "
        f"to_download={len(unique_jobs)} alias_pending={len(alias_jobs)}"
    )

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _download_one,
                sid,
                vid,
                info,
                audio_dir,
                yt_dlp=yt_dlp,
                cookies_from_browser=args.cookies_from_browser,
                sleep_min=args.sleep_interval,
                sleep_max=args.max_sleep_interval,
                retries=args.retries,
                concurrent_fragments=args.concurrent_fragments,
            )
            for sid, vid, info in unique_jobs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            sid, vid, yid, path, err = fut.result()
            if path is not None:
                path_by_yid[yid] = path
                ok_by_key[(sid, vid)] = path
            else:
                n_fail += 1
                fail_lines.append(f"{sid}\t{vid}\t{yid}\t{err}")

    for sid, vid, info in alias_jobs:
        yid = info["youtube_id"]
        path = path_by_yid.get(yid)
        if path is not None:
            ok_by_key[(sid, vid)] = path
        else:
            n_fail += 1
            fail_lines.append(f"{sid}\t{vid}\t{yid}\talias of failed download")

    fail_log.write_text(
        "\n".join(fail_lines) + ("\n" if fail_lines else ""), encoding="utf-8"
    )
    print(
        f"downloaded_ok={len(ok_by_key)} resume={n_resume} fail={n_fail} log={fail_log}"
    )

    by_work: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for (sid, vid), path in ok_by_key.items():
        by_work[sid].append((vid, path))
    cliques = {sid: tracks for sid, tracks in by_work.items() if len(tracks) >= 2}
    print(
        f"cliques_kept={len(cliques)} tracks_in_cliques="
        f"{sum(len(v) for v in cliques.values())} "
        f"(dropped_works={len(by_work) - len(cliques)})"
    )
    if not cliques:
        raise SystemExit("No cliques with ≥2 downloaded tracks; nothing to package.")

    corpus = {"id": [], "audio": []}
    queries = {"id": [], "audio": []}
    qrels = {"query-id": [], "corpus-id": [], "score": []}

    for sid, tracks in tqdm(sorted(cliques.items()), desc="build"):
        ids: list[str] = []
        paths: list[str] = []
        for vid, path in sorted(
            tracks, key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
        ):
            cid = f"{sid}__{vid}"
            corpus["id"].append(cid)
            corpus["audio"].append(str(path))
            ids.append(cid)
            paths.append(str(path))
        for qid, qpath in zip(ids, paths, strict=True):
            query_id = f"q-{qid}"
            queries["id"].append(query_id)
            queries["audio"].append(qpath)
            for tid in ids:
                if tid == qid:
                    continue
                qrels["query-id"].append(query_id)
                qrels["corpus-id"].append(tid)
                qrels["score"].append(1)

    print(
        f"corpus={len(corpus['id'])} queries={len(queries['id'])} "
        f"qrels={len(qrels['query-id'])}"
    )

    corpus_ds = Dataset.from_dict(corpus).cast_column("audio", Audio())
    queries_ds = Dataset.from_dict(queries).cast_column("audio", Audio())
    qrels_ds = Dataset.from_dict(qrels)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus_ds}).push_to_hub(
            args.repo_id, "corpus", token=token
        )
        DatasetDict({"test": queries_ds}).push_to_hub(
            args.repo_id, "queries", token=token
        )
        DatasetDict({"test": qrels_ds}).push_to_hub(args.repo_id, "qrels", token=token)
        from huggingface_hub import dataset_info

        sha = dataset_info(args.repo_id, token=token).sha
        rev_path = work / "hub_revision.txt"
        rev_path.write_text(sha + "\n", encoding="utf-8")
        print(
            f"Pushed {args.repo_id} @ {sha}\n"
            f"Pin this SHA in mteb/tasks/retrieval/eng/shs100k_retrieval.py "
            f"(also written to {rev_path})."
        )
    else:
        out = work / "mteb_export"
        out.mkdir(parents=True, exist_ok=True)
        corpus_ds.save_to_disk(out / "corpus")
        queries_ds.save_to_disk(out / "queries")
        qrels_ds.save_to_disk(out / "qrels")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
