#!/usr/bin/env python3
"""Construct the availability-frozen FIVR-5K MTEB metadata dataset.

The public Hugging Face artifact produced by this script contains identifiers,
source URLs, availability decisions, and qrels. It deliberately does not upload
the underlying YouTube video bytes. With ``--download-media``, the same script
can materialize a local, resumable media cache for validation and evaluation.

Example:
    uvx --with av --with datasets --with huggingface-hub --with pytubefix \
        --with yt-dlp python \
        scripts/data/fivr/create_data.py \
        --fivr-dir /path/to/FIVR-200K \
        --videoeval-dir /path/to/VideoEval \
        --visil-pickle /path/to/visil/datasets/fivr.pickle \
        --availability-audit availability-2026-08-10.json \
        --output-dir /tmp/fivr-mteb \
        --video-dir /tmp/fivr-mteb/videos --download-media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

FIVR_REVISION = "77e3d8a9c159eed9a0b9686e2d62bc7ab3106e24"
VIDEOEVAL_REVISION = "8b86d707ce65ab07323745f7b9669e0d404e11e0"
VISIL_REVISION = "0971e54fb8325fceb1bc9748ecbfe4c66e5dabd2"

EXPECTED_FILE_SHA256 = {
    "fivr_annotation": (
        "ccd343bd1c442142da3b2ef9e571ae888d44ac039fd02de708b0be820385a3ea"
    ),
    "fivr_ids": ("6fa192644eb84f4a3b88839d850de9d38ac25de90622ccb8cf4a111a75829c06"),
    "videoeval_annotation": (
        "ac75b018fad02bdfba233f47bc1c561542309d8b91728ca6dc63f54a164f49ee"
    ),
    "videoeval_queries": (
        "f98302ab962d5da4c84e52cd0c7670f1b125882052894250efd46578381f43bd"
    ),
    "videoeval_database": (
        "ac32d2b863580fdf60b7b62da67d9d2f98631f4071d00c889a282865625a96a2"
    ),
    "videoeval_manifest": (
        "b2e6718db1f68ed70cfb485c6134149de618d045caae8e7b0940bf64899da8ec"
    ),
    "visil_metadata": (
        "05664821a6f3f5ccdb0cbfbcd26dd712e6aea64f9a1665abe0743110d04e991e"
    ),
}

LABELS = ("ND", "DS", "CS", "IS")
REGIMES = {
    "dsvr": ("ND", "DS"),
    "csvr": ("ND", "DS", "CS"),
    "isvr": ("ND", "DS", "CS", "IS"),
}
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


class _RestrictedUnpickler(pickle.Unpickler):
    """Load ViSiL's primitive-only metadata pickle without arbitrary globals."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("builtins", "set"):
            return set
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


class _QuietYtdlpLogger:
    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def _load_ids(path: Path) -> list[str]:
    return [
        line.split()[0].removesuffix(".mp4")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalise_annotation(annotation: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    return {
        query_id: {label: set(labels.get(label, [])) for label in LABELS}
        for query_id, labels in annotation.items()
    }


def _load_visil(path: Path) -> dict[str, Any]:
    _validate_sha(path, EXPECTED_FILE_SHA256["visil_metadata"], "ViSiL metadata")
    with path.open("rb") as stream:
        return cast("dict[str, Any]", _RestrictedUnpickler(stream).load())


def _source_paths(
    fivr_dir: Path, videoeval_dir: Path
) -> tuple[Path, Path, Path, Path, Path, Path]:
    fivr_dataset = fivr_dir / "dataset"
    videoeval = videoeval_dir / "VidEB" / "annotations" / "FIVR-5K"
    return (
        fivr_dataset / "annotation.json",
        fivr_dataset / "youtube_ids.txt",
        videoeval / "annotation.json",
        videoeval / "queries" / "test.csv",
        videoeval / "database" / "test.csv",
        videoeval / "used_videos.txt",
    )


def _validate_sources(
    *,
    fivr_dir: Path,
    videoeval_dir: Path,
    visil_pickle: Path,
) -> dict[str, Any]:
    (
        fivr_annotation_path,
        fivr_ids_path,
        videoeval_annotation_path,
        query_path,
        database_path,
        manifest_path,
    ) = _source_paths(fivr_dir, videoeval_dir)

    for path in (
        fivr_annotation_path,
        fivr_ids_path,
        videoeval_annotation_path,
        query_path,
        database_path,
        manifest_path,
        visil_pickle,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for key, path in (
        ("fivr_annotation", fivr_annotation_path),
        ("fivr_ids", fivr_ids_path),
        ("videoeval_annotation", videoeval_annotation_path),
        ("videoeval_queries", query_path),
        ("videoeval_database", database_path),
        ("videoeval_manifest", manifest_path),
    ):
        _validate_sha(path, EXPECTED_FILE_SHA256[key], key)

    fivr_ids = _load_ids(fivr_ids_path)
    official_annotation = json.loads(fivr_annotation_path.read_text(encoding="utf-8"))
    videoeval_annotation = json.loads(
        videoeval_annotation_path.read_text(encoding="utf-8")
    )
    if official_annotation != videoeval_annotation:
        raise ValueError("VideoEval annotation.json differs from FIVR-200K")
    if len(fivr_ids) != 225_960 or len(set(fivr_ids)) != len(fivr_ids):
        raise ValueError("unexpected FIVR-200K ID count or duplicate ID")
    if len(official_annotation) != 100:
        raise ValueError("unexpected FIVR-200K query annotation count")

    queries = _load_ids(query_path)
    database = _load_ids(database_path)
    manifest_rows = _load_ids(manifest_path)
    manifest_ids = list(dict.fromkeys(manifest_rows))
    if len(queries) != 31 or len(database) != 3415:
        raise ValueError("unexpected VideoEval FIVR-5K query/database counts")
    if len(set(queries)) != len(queries) or len(set(database)) != len(database):
        raise ValueError("duplicate ID within VideoEval query or database rows")
    if len(manifest_rows) != 3446 or len(manifest_ids) != 3445:
        raise ValueError("unexpected VideoEval FIVR-5K manifest counts")
    if set(queries) | set(database) != set(manifest_ids):
        raise ValueError("VideoEval query/database union differs from used_videos.txt")
    if set(queries) & set(database) != {"eCrhXArKE24"}:
        raise ValueError("unexpected VideoEval query/database overlap")
    if not set(manifest_ids) <= set(fivr_ids):
        raise ValueError("VideoEval manifest contains IDs outside FIVR-200K")
    if not set(queries) <= set(official_annotation):
        raise ValueError("VideoEval query is missing from official annotations")
    cross_query_positives = {
        (query_id, video_id)
        for query_id in queries
        for label in LABELS
        for video_id in official_annotation[query_id].get(label, [])
        if video_id in set(queries) - {query_id}
    }
    if cross_query_positives:
        raise ValueError(
            "removing query IDs from the corpus would remove cross-query positives"
        )

    visil = _load_visil(visil_pickle)
    canonical_queries = list(visil["5k"]["queries"])
    canonical_database = set(visil["5k"]["database"])
    if len(canonical_queries) != 50 or len(canonical_database) != 5000:
        raise ValueError("unexpected canonical ViSiL FIVR-5K counts")
    if not set(queries) <= set(canonical_queries):
        raise ValueError("VideoEval queries are not a subset of canonical FIVR-5K")
    if not set(database) <= canonical_database:
        raise ValueError("VideoEval database is not a subset of canonical FIVR-5K")
    visil_annotation = _normalise_annotation(visil["annotation"])
    current_annotation = _normalise_annotation(official_annotation)
    if set(visil_annotation) != set(current_annotation):
        raise ValueError("ViSiL and current FIVR annotations cover different queries")
    changed_queries = sorted(
        query_id
        for query_id in current_annotation
        if visil_annotation[query_id] != current_annotation[query_id]
    )
    changed_videoeval_queries = sorted(set(changed_queries) & set(queries))
    manifest_set = set(manifest_ids)
    manifest_differences = sorted(
        (query_id, label, video_id, "visil-only")
        for query_id in queries
        for label in LABELS
        for video_id in (
            visil_annotation[query_id][label] - current_annotation[query_id][label]
        )
        if video_id in manifest_set
    ) + sorted(
        (query_id, label, video_id, "current-only")
        for query_id in queries
        for label in LABELS
        for video_id in (
            current_annotation[query_id][label] - visil_annotation[query_id][label]
        )
        if video_id in manifest_set
    )
    expected_manifest_differences = [
        ("5MBA_7vDhII", "IS", "Z8ZhfkHyTmQ", "visil-only"),
        ("e33dD_qoqhg", "IS", "wcVRzMqpfCg", "visil-only"),
    ]
    if manifest_differences != expected_manifest_differences:
        raise ValueError(
            "unexpected annotation drift between ViSiL and current FIVR metadata"
        )

    return {
        "annotation": official_annotation,
        "fivr_ids": fivr_ids,
        "queries": queries,
        "database": database,
        "manifest_rows": manifest_rows,
        "manifest_ids": manifest_ids,
        "canonical_queries": canonical_queries,
        "canonical_database": canonical_database,
        "annotation_drift": {
            "changed_queries": len(changed_queries),
            "changed_videoeval_queries": len(changed_videoeval_queries),
            "manifest_differences": [
                {
                    "query_id": query_id,
                    "label": label,
                    "video_id": video_id,
                    "direction": direction,
                }
                for query_id, label, video_id, direction in manifest_differences
            ],
        },
    }


def _load_audit(path: Path, manifest_ids: list[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("manifest", {}).get("sha256")
        != EXPECTED_FILE_SHA256["videoeval_manifest"]
    ):
        raise ValueError("availability audit refers to a different manifest")
    results = payload.get("results", [])
    by_id = {row["id"]: row for row in results}
    if set(by_id) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(by_id))
        extra = sorted(set(by_id) - set(manifest_ids))
        raise ValueError(
            f"availability audit is incomplete: missing={missing[:5]} extra={extra[:5]}"
        )
    transient = {
        row["id"]
        for row in results
        if row["status"] in {"bot_check", "error", "rate_limited"}
    }
    if transient:
        raise ValueError(
            "availability audit contains unresolved transient statuses: "
            + ", ".join(sorted(transient)[:10])
        )
    return by_id


def _ordered_positives(
    annotation: dict[str, Any], query_id: str, labels: tuple[str, ...]
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        for video_id in annotation[query_id].get(label, []):
            if video_id not in seen:
                result.append(video_id)
                seen.add(video_id)
    return result


def _build_rows(
    sources: dict[str, Any], audit: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    annotation = sources["annotation"]
    videoeval_queries = sources["queries"]
    videoeval_database = sources["database"]
    query_set = set(videoeval_queries)

    # VideoEval includes one query in its database. Its evaluator explicitly
    # skips self-matches, so remove every query ID from the MTEB corpus.
    original_corpus = [
        video_id for video_id in videoeval_database if video_id not in query_set
    ]
    available_ids = {video_id for video_id, row in audit.items() if row["available"]}
    corpus_ids = [video_id for video_id in original_corpus if video_id in available_ids]
    corpus_set = set(corpus_ids)

    query_decisions: list[dict[str, Any]] = []
    retained_queries: list[str] = []
    for query_id in videoeval_queries:
        strict_positives = (
            set(_ordered_positives(annotation, query_id, REGIMES["dsvr"])) & corpus_set
        )
        query_available = query_id in available_ids
        retained = query_available and bool(strict_positives)
        if not query_available:
            reason = f"query media is {audit[query_id]['status']}"
        elif not strict_positives:
            reason = "no surviving DSVR positive"
        else:
            reason = "retained"
            retained_queries.append(query_id)
        query_decisions.append(
            {
                "id": query_id,
                "availability_status": audit[query_id]["status"],
                "surviving_dsvr_positives": len(strict_positives),
                "retained": retained,
                "reason": reason,
            }
        )

    corpus_rows = [
        {
            "id": video_id,
            "youtube_id": video_id,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "availability_status": audit[video_id]["status"],
            "duration_seconds": audit[video_id].get("duration_seconds"),
        }
        for video_id in corpus_ids
    ]
    query_rows = [
        {
            "id": video_id,
            "youtube_id": video_id,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "availability_status": audit[video_id]["status"],
            "duration_seconds": audit[video_id].get("duration_seconds"),
        }
        for video_id in retained_queries
    ]

    qrels: dict[str, list[dict[str, Any]]] = {}
    for regime, labels in REGIMES.items():
        rows: list[dict[str, Any]] = []
        for query_id in retained_queries:
            positives = _ordered_positives(annotation, query_id, labels)
            rows.extend(
                {"query-id": query_id, "corpus-id": corpus_id, "score": 1}
                for corpus_id in positives
                if corpus_id in corpus_set
            )
        qrels[regime] = rows

    role_by_id: dict[str, set[str]] = {}
    for video_id in videoeval_queries:
        role_by_id.setdefault(video_id, set()).add("query")
    for video_id in videoeval_database:
        role_by_id.setdefault(video_id, set()).add("database")
    availability_rows = [
        {
            "id": video_id,
            "manifest_role": "+".join(sorted(role_by_id[video_id])),
            "available": bool(audit[video_id]["available"]),
            "status": audit[video_id]["status"],
            "duration_seconds": audit[video_id].get("duration_seconds"),
        }
        for video_id in sources["manifest_ids"]
    ]

    loss_rows: list[dict[str, Any]] = []
    original_corpus_set = set(original_corpus)
    for query_id in videoeval_queries:
        for label in LABELS:
            original = [
                video_id
                for video_id in annotation[query_id].get(label, [])
                if video_id in original_corpus_set
            ]
            surviving = [video_id for video_id in original if video_id in corpus_set]
            lost = [video_id for video_id in original if video_id not in corpus_set]
            loss_rows.append(
                {
                    "query-id": query_id,
                    "label": label,
                    "original_positives": len(original),
                    "surviving_positives": len(surviving),
                    "lost_positives": len(lost),
                    "lost_ids": lost,
                    "lost_statuses": [audit[video_id]["status"] for video_id in lost],
                }
            )

    return {
        "corpus": corpus_rows,
        "queries": query_rows,
        "qrels": qrels,
        "availability": availability_rows,
        "positive_losses": loss_rows,
        "query_decisions": query_decisions,
    }


def _validate_rows(rows: dict[str, Any]) -> None:
    corpus_ids = {row["id"] for row in rows["corpus"]}
    query_ids = {row["id"] for row in rows["queries"]}
    if len(corpus_ids) != len(rows["corpus"]):
        raise ValueError("duplicate corpus IDs")
    if len(query_ids) != len(rows["queries"]):
        raise ValueError("duplicate query IDs")
    if corpus_ids & query_ids:
        raise ValueError("query IDs must not appear in the corpus")
    for regime, qrel_rows in rows["qrels"].items():
        qrel_queries = {row["query-id"] for row in qrel_rows}
        if qrel_queries != query_ids:
            raise ValueError(f"{regime} qrels do not cover every retained query")
        for row in qrel_rows:
            if row["corpus-id"] not in corpus_ids:
                raise ValueError(f"{regime} qrel references missing corpus media")
            if row["score"] != 1:
                raise ValueError(f"{regime} contains a non-binary qrel")
    qrel_pairs = {
        regime: {(row["query-id"], row["corpus-id"]) for row in qrel_rows}
        for regime, qrel_rows in rows["qrels"].items()
    }
    if not qrel_pairs["dsvr"] <= qrel_pairs["csvr"] <= qrel_pairs["isvr"]:
        raise ValueError("official DSVR/CSVR/ISVR relevance unions are not nested")


def _find_video(video_dir: Path, video_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _video_decodes(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    try:
        import av

        with av.open(str(path)) as container:
            return next(container.decode(video=0), None) is not None
    except ImportError:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _download_one(
    video_id: str,
    video_dir: Path,
    resolution: int,
    force_ipv4: bool,
    backend: str,
) -> dict[str, Any]:
    existing = _find_video(video_dir, video_id)
    if existing is not None and _video_decodes(existing):
        return {
            "id": video_id,
            "ok": True,
            "path": str(existing),
            "resumed": True,
            "backend": "cache",
        }
    if existing is not None:
        existing.unlink()

    errors: list[str] = []
    if backend in {"auto", "yt-dlp"}:
        try:
            import yt_dlp  # type: ignore[import-untyped]

            options: dict[str, Any] = {
                "continuedl": True,
                "format": (
                    f"bestvideo[height<={resolution}]/best[height<={resolution}]/"
                    "bestvideo/best"
                ),
                "logger": _QuietYtdlpLogger(),
                "noplaylist": True,
                "no_warnings": True,
                "outtmpl": str(video_dir / f"{video_id}.%(ext)s"),
                "quiet": True,
                "retries": 3,
            }
            if force_ipv4:
                options["source_address"] = "0.0.0.0"
            node = shutil.which("node")
            if node is not None:
                options["js_runtimes"] = {"node": {"path": node}}
                options["remote_components"] = {"ejs:github"}
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
            path = _find_video(video_dir, video_id)
            if path is None or not _video_decodes(path):
                raise RuntimeError("downloaded file is missing or undecodable")
            return {
                "id": video_id,
                "ok": True,
                "path": str(path),
                "resumed": False,
                "backend": "yt-dlp",
            }
        except Exception as error:
            errors.append(f"yt-dlp: {error}")
            failed_path = _find_video(video_dir, video_id)
            if failed_path is not None and not _video_decodes(failed_path):
                failed_path.unlink()

    try:
        from pytubefix import YouTube  # type: ignore[import-untyped]

        video = YouTube(
            f"https://www.youtube.com/watch?v={video_id}",
            use_oauth=False,
            allow_oauth_cache=False,
        )
        stream_groups = (
            list(video.streams.filter(progressive=True, file_extension="mp4")),
            list(video.streams.filter(only_video=True, file_extension="mp4")),
            list(video.streams.filter(progressive=True)),
            list(video.streams.filter(only_video=True)),
        )
        streams = next((group for group in stream_groups if group), [])
        eligible = [
            stream
            for stream in streams
            if stream.resolution is not None
            and int(stream.resolution.removesuffix("p")) <= resolution
        ]
        stream = max(
            eligible or streams,
            key=lambda item: int((item.resolution or "0p").removesuffix("p")),
            default=None,
        )
        if stream is None:
            raise RuntimeError("no downloadable video stream")
        suffix = f".{stream.subtype or 'mp4'}"
        path = Path(
            stream.download(output_path=video_dir, filename=f"{video_id}{suffix}")
        )
        if not _video_decodes(path):
            raise RuntimeError("downloaded file is undecodable")
        return {
            "id": video_id,
            "ok": True,
            "path": str(path),
            "resumed": False,
            "backend": "pytubefix",
        }
    except Exception as error:
        for partial in video_dir.glob(f"{video_id}.*"):
            if partial.suffix in VIDEO_SUFFIXES and not _video_decodes(partial):
                partial.unlink()
        errors.append(f"pytubefix: {error}")
        return {"id": video_id, "ok": False, "error": "; ".join(errors)[-1000:]}


def _materialize_media(
    *,
    rows: dict[str, Any],
    video_dir: Path,
    workers: int,
    resolution: int,
    force_ipv4: bool,
    backend: str,
    output_dir: Path,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    ids = list(
        dict.fromkeys(
            [row["id"] for row in rows["queries"]]
            + [row["id"] for row in rows["corpus"]]
        )
    )
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _download_one,
                video_id,
                video_dir,
                resolution,
                force_ipv4,
                backend,
            ): video_id
            for video_id in ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["id"]] = result
            if completed % 50 == 0:
                print(f"media {completed}/{len(ids)}", flush=True)

    ordered = [results[video_id] for video_id in ids]
    report = {
        "requested": len(ids),
        "successful": sum(row["ok"] for row in ordered),
        "failed": sum(not row["ok"] for row in ordered),
        "resolution": resolution,
        "force_ipv4": force_ipv4,
        "requested_backend": backend,
        "results": ordered,
    }
    _write_json(output_dir / "download-report.json", report)
    failures = [row for row in ordered if not row["ok"]]
    if failures:
        raise RuntimeError(
            f"{len(failures)} frozen videos failed to download; see download-report.json"
        )


def _summary(sources: dict[str, Any], rows: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows["availability"])

    def aggregate_losses(query_ids: set[str] | None = None) -> dict[str, Any]:
        selected = [
            row
            for row in rows["positive_losses"]
            if query_ids is None or row["query-id"] in query_ids
        ]
        return {
            "by_label": {
                label: {
                    "original_positives": sum(
                        row["original_positives"]
                        for row in selected
                        if row["label"] == label
                    ),
                    "surviving_positives": sum(
                        row["surviving_positives"]
                        for row in selected
                        if row["label"] == label
                    ),
                    "lost_positives": sum(
                        row["lost_positives"]
                        for row in selected
                        if row["label"] == label
                    ),
                }
                for label in LABELS
            },
            "affected_queries": len(
                {row["query-id"] for row in selected if row["lost_positives"]}
            ),
        }

    retained_query_ids = {row["id"] for row in rows["queries"]}
    qrel_stats: dict[str, Any] = {}
    for regime, qrel_rows in rows["qrels"].items():
        per_query = Counter(row["query-id"] for row in qrel_rows)
        counts = list(per_query.values())
        qrel_stats[regime] = {
            "qrels": len(qrel_rows),
            "min_per_query": min(counts),
            "mean_per_query": sum(counts) / len(counts),
            "max_per_query": max(counts),
        }
    return {
        "source_revisions": {
            "fivr_200k": FIVR_REVISION,
            "videoeval": VIDEOEVAL_REVISION,
            "visil": VISIL_REVISION,
        },
        "canonical_fivr5k": {
            "queries": len(sources["canonical_queries"]),
            "database": len(sources["canonical_database"]),
        },
        "videoeval_fivr5k": {
            "queries": len(sources["queries"]),
            "database_rows": len(sources["database"]),
            "manifest_rows": len(sources["manifest_rows"]),
            "manifest_unique_ids": len(sources["manifest_ids"]),
            "annotation_drift_from_visil_release": sources["annotation_drift"],
        },
        "availability": {
            "available": sum(row["available"] for row in rows["availability"]),
            "unavailable": sum(not row["available"] for row in rows["availability"]),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "positive_losses": {
            "all_videoeval_queries": aggregate_losses(),
            "retained_mteb_queries": aggregate_losses(retained_query_ids),
        },
        "mteb": {
            "queries": len(rows["queries"]),
            "corpus": len(rows["corpus"]),
            "qrels": qrel_stats,
            "dropped_queries": [
                row for row in rows["query_decisions"] if not row["retained"]
            ],
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_artifacts(
    output_dir: Path, rows: dict[str, Any], summary: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus.jsonl", rows["corpus"])
    _write_jsonl(output_dir / "queries.jsonl", rows["queries"])
    for regime, qrels in rows["qrels"].items():
        _write_jsonl(output_dir / f"{regime}-qrels.jsonl", qrels)
    _write_jsonl(output_dir / "availability.jsonl", rows["availability"])
    _write_jsonl(output_dir / "positive-losses.jsonl", rows["positive_losses"])
    _write_jsonl(output_dir / "query-decisions.jsonl", rows["query_decisions"])
    _write_json(output_dir / "summary.json", summary)


def _render_card(template_path: Path, summary: dict[str, Any]) -> str:
    mteb = summary["mteb"]
    availability = summary["availability"]
    all_losses = summary["positive_losses"]["all_videoeval_queries"]["by_label"]
    retained_losses = summary["positive_losses"]["retained_mteb_queries"]["by_label"]
    replacements = {
        "{corpus}": mteb["corpus"],
        "{queries}": mteb["queries"],
        "{available}": availability["available"],
        "{unavailable}": availability["unavailable"],
        "{dsvr_qrels}": mteb["qrels"]["dsvr"]["qrels"],
        "{csvr_qrels}": mteb["qrels"]["csvr"]["qrels"],
        "{isvr_qrels}": mteb["qrels"]["isvr"]["qrels"],
        "{all_nd_lost}": all_losses["ND"]["lost_positives"],
        "{all_ds_lost}": all_losses["DS"]["lost_positives"],
        "{all_cs_lost}": all_losses["CS"]["lost_positives"],
        "{all_is_lost}": all_losses["IS"]["lost_positives"],
        "{retained_nd_lost}": retained_losses["ND"]["lost_positives"],
        "{retained_ds_lost}": retained_losses["DS"]["lost_positives"],
        "{retained_cs_lost}": retained_losses["CS"]["lost_positives"],
        "{retained_is_lost}": retained_losses["IS"]["lost_positives"],
    }
    card = template_path.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        card = card.replace(placeholder, str(value))
    return card


def _push(
    *,
    repo_id: str,
    rows: dict[str, Any],
    output_dir: Path,
    card: str,
) -> str:
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    configs = {
        "corpus": rows["corpus"],
        "queries": rows["queries"],
        "dsvr-qrels": rows["qrels"]["dsvr"],
        "csvr-qrels": rows["qrels"]["csvr"],
        "isvr-qrels": rows["qrels"]["isvr"],
        "availability": rows["availability"],
        "positive-losses": rows["positive_losses"],
        "query-decisions": rows["query_decisions"],
    }
    for config, config_rows in configs.items():
        DatasetDict({"test": Dataset.from_list(config_rows)}).push_to_hub(
            repo_id, config
        )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document FIVR-5K MTEB metadata",
    )
    for filename in ("summary.json", "availability.jsonl", "positive-losses.jsonl"):
        api.upload_file(
            path_or_fileobj=output_dir / filename,
            path_in_repo=f"audit/{filename}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {filename}",
        )
    revision = api.dataset_info(repo_id).sha
    if revision is None:
        raise RuntimeError("Hugging Face did not return an immutable revision")
    (output_dir / "hub-revision.txt").write_text(revision + "\n", encoding="utf-8")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fivr-dir", type=Path, required=True)
    parser.add_argument("--videoeval-dir", type=Path, required=True)
    parser.add_argument("--visil-pickle", type=Path, required=True)
    parser.add_argument(
        "--availability-audit",
        type=Path,
        default=Path(__file__).with_name("availability-2026-08-10.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument(
        "--download-backend",
        choices=("auto", "yt-dlp", "pytubefix"),
        default="auto",
    )
    parser.add_argument("--force-ipv4", action="store_true")
    parser.add_argument("--repo-id", default="Cerru02/FIVR-5K-MTEB")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    sources = _validate_sources(
        fivr_dir=args.fivr_dir,
        videoeval_dir=args.videoeval_dir,
        visil_pickle=args.visil_pickle,
    )
    audit = _load_audit(args.availability_audit, sources["manifest_ids"])
    rows = _build_rows(sources, audit)
    _validate_rows(rows)
    summary = _summary(sources, rows)
    _write_artifacts(args.output_dir, rows, summary)

    if args.download_media:
        if args.video_dir is None:
            raise ValueError("--video-dir is required with --download-media")
        _materialize_media(
            rows=rows,
            video_dir=args.video_dir,
            workers=args.workers,
            resolution=args.resolution,
            force_ipv4=args.force_ipv4,
            backend=args.download_backend,
            output_dir=args.output_dir,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.push:
        card_template = Path(__file__).with_name("dataset_card.md")
        revision = _push(
            repo_id=args.repo_id,
            rows=rows,
            output_dir=args.output_dir,
            card=_render_card(card_template, summary),
        )
        print(f"Pushed {args.repo_id} @ {revision}")


if __name__ == "__main__":
    main()
