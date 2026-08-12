"""Audit availability of the frozen VideoEval FIVR-5K YouTube manifest.

The script never downloads media. It resolves each YouTube ID with yt-dlp or
pytubefix, records a structured status, and writes results in manifest order.
The output is intended to be frozen and consumed by ``create_data.py``;
construction must not silently reinterpret a later availability state.

Example:
    uvx --with yt-dlp python scripts/data/fivr/audit_availability.py \
        --manifest VidEB/annotations/FIVR-5K/used_videos.txt \
        --output availability-2026-08-10.json --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class _QuietLogger:
    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def _classify_error(message: str) -> str:
    message_lower = message.lower()
    patterns = (
        ("private video", "private"),
        ("copyright", "copyright_blocked"),
        ("removed", "removed"),
        ("no longer available", "removed"),
        ("video unavailable", "unavailable"),
        ("not available", "unavailable"),
        ("confirm your age", "age_restricted"),
        ("age-restricted", "age_restricted"),
        ("members-only", "members_only"),
        ("members only", "members_only"),
        ("not available in your country", "geo_blocked"),
        ("not made this video available in your country", "geo_blocked"),
        ("sign in to confirm you’re not a bot", "bot_check"),
        ("sign in to confirm you're not a bot", "bot_check"),
        ("too many requests", "rate_limited"),
        ("http error 429", "rate_limited"),
        ("please sign in", "login_required"),
    )
    for pattern, status in patterns:
        if pattern in message_lower:
            return status
    return "error"


def _load_ytdlp() -> Any:
    try:
        return importlib.import_module("yt_dlp")
    except ImportError as error:
        raise RuntimeError(
            "The yt-dlp backend requires the optional 'yt-dlp' package. "
            "Install it or run with --backend pytubefix."
        ) from error


def _probe_ytdlp(
    video_id: str,
    socket_timeout: int,
    request_delay: float,
    force_ipv4: bool,
) -> dict[str, Any]:
    yt_dlp = _load_ytdlp()
    if request_delay:
        time.sleep(request_delay)
    options = {
        "extractor_retries": 1,
        "fragment_retries": 0,
        "noplaylist": True,
        "no_warnings": True,
        "quiet": True,
        "logger": _QuietLogger(),
        "retries": 1,
        "skip_download": True,
        "socket_timeout": socket_timeout,
    }
    if force_ipv4:
        options["source_address"] = "0.0.0.0"
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        if info is None:
            return {
                "id": video_id,
                "available": False,
                "status": "error",
                "probe_backend": "yt-dlp",
            }
        return {
            "id": video_id,
            "available": True,
            "status": info.get("availability") or "available",
            "duration_seconds": info.get("duration"),
            "probe_backend": "yt-dlp",
        }
    except Exception as error:  # yt-dlp exposes several extractor exceptions
        message = str(error)
        return {
            "id": video_id,
            "available": False,
            "status": _classify_error(message),
            "error": message,
            "probe_backend": "yt-dlp",
        }


def _probe_pytubefix(video_id: str, request_delay: float) -> dict[str, Any]:
    if request_delay:
        time.sleep(request_delay)
    try:
        pytubefix = importlib.import_module("pytubefix")

        video = pytubefix.YouTube(
            f"https://www.youtube.com/watch?v={video_id}",
            use_oauth=False,
            allow_oauth_cache=False,
        )
        duration = video.length
        if not video.streams:
            raise RuntimeError("no downloadable streams")
        return {
            "id": video_id,
            "available": True,
            "status": "public",
            "duration_seconds": duration,
            "probe_backend": "pytubefix",
        }
    except Exception as error:
        error_type = type(error).__name__
        type_statuses = {
            "AccountTerminated": "removed",
            "AgeRestrictedError": "age_restricted",
            "BotDetection": "bot_check",
            "LiveStreamError": "unavailable",
            "MembersOnly": "members_only",
            "PrivateVideo": "private",
            "RecordingUnavailable": "unavailable",
            "VideoRegionBlocked": "geo_blocked",
            "VideoUnavailable": "unavailable",
        }
        message = str(error)
        return {
            "id": video_id,
            "available": False,
            "status": type_statuses.get(error_type, _classify_error(message)),
            "error": f"{error_type}: {message}",
            "probe_backend": "pytubefix",
        }


def _probe(
    video_id: str,
    socket_timeout: int,
    request_delay: float,
    force_ipv4: bool,
    backend: str,
) -> dict[str, Any]:
    if backend == "pytubefix":
        return _probe_pytubefix(video_id, request_delay)
    return _probe_ytdlp(video_id, socket_timeout, request_delay, force_ipv4)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove verbose extractor errors while preserving every audit decision."""
    public = {key: value for key, value in payload.items() if key != "results"}
    public["manifest"] = {
        **public["manifest"],
        "path": "VidEB/annotations/FIVR-5K/used_videos.txt",
    }
    backends = sorted(
        {result.get("probe_backend", "yt-dlp") for result in payload["results"]}
    )
    public["probe"] = {
        "method": "multi-pass stream enumeration",
        "backends": {
            backend: importlib.metadata.version(backend) for backend in backends
        },
    }
    public["network"] = {
        "note": (
            "IPv4 was used for yt-dlp retries; final transient statuses were "
            "rechecked serially with a request delay."
        )
    }
    public["results"] = [
        {key: value for key, value in result.items() if key != "error"}
        for result in payload["results"]
    ]
    return public


def _payload(
    *,
    manifest: Path,
    manifest_ids: list[str],
    results: dict[str, dict[str, Any]],
    checked_at: str,
    force_ipv4: bool,
    backend: str,
) -> dict[str, Any]:
    unique_ids = list(dict.fromkeys(manifest_ids))
    ordered_results = [
        results[video_id] for video_id in unique_ids if video_id in results
    ]
    statuses = Counter(result["status"] for result in ordered_results)
    return {
        "schema_version": 1,
        "checked_at_utc": checked_at,
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "probe": {
            "backend": backend,
            "version": importlib.metadata.version(backend),
        },
        "network": {"force_ipv4": force_ipv4},
        "manifest": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "rows": len(manifest_ids),
            "unique_ids": len(unique_ids),
            "duplicate_ids": sorted(
                video_id
                for video_id, count in Counter(manifest_ids).items()
                if count > 1
            ),
        },
        "summary": {
            "checked": len(ordered_results),
            "available": sum(result["available"] for result in ordered_results),
            "unavailable": sum(not result["available"] for result in ordered_results),
            "status_counts": dict(sorted(statuses.items())),
        },
        "results": ordered_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-output",
        type=Path,
        help="Also write a compact artifact without verbose extractor errors.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--socket-timeout", type=int, default=20)
    parser.add_argument(
        "--backend",
        choices=("yt-dlp", "pytubefix"),
        default="yt-dlp",
    )
    parser.add_argument(
        "--force-ipv4",
        action="store_true",
        help="Use IPv4, useful when an IPv6 egress is classified as a bot.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0,
        help="Delay each probe to reduce the risk of remote rate limiting.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Atomically refresh the output after this many completed probes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse IDs already present in the output file.",
    )
    parser.add_argument(
        "--retry-status",
        action="append",
        default=[],
        help="When resuming, probe this prior status again (repeatable).",
    )
    args = parser.parse_args()

    if args.backend == "yt-dlp":
        _load_ytdlp()

    manifest_ids = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique_ids = list(dict.fromkeys(manifest_ids))
    results: dict[str, dict[str, Any]] = {}
    checked_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        retry_statuses = set(args.retry_status)
        previous_results = previous.get("results", [])
        for result in previous_results:
            result.setdefault("probe_backend", "yt-dlp")
        results = {
            result["id"]: result
            for result in previous_results
            if result["status"] not in retry_statuses
        }
        checked_at = previous.get("checked_at_utc", checked_at)

    pending = [video_id for video_id in unique_ids if video_id not in results]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _probe,
                video_id,
                args.socket_timeout,
                args.request_delay,
                args.force_ipv4,
                args.backend,
            ): video_id
            for video_id in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["id"]] = result
            if completed % args.checkpoint_every == 0:
                payload = _payload(
                    manifest=args.manifest,
                    manifest_ids=manifest_ids,
                    results=results,
                    checked_at=checked_at,
                    force_ipv4=args.force_ipv4,
                    backend=args.backend,
                )
                _write_json(args.output, payload)
                print(
                    f"checked {len(results)}/{len(unique_ids)}: "
                    f"{payload['summary']['status_counts']}",
                    flush=True,
                )

    payload = _payload(
        manifest=args.manifest,
        manifest_ids=manifest_ids,
        results=results,
        checked_at=checked_at,
        force_ipv4=args.force_ipv4,
        backend=args.backend,
    )
    _write_json(args.output, payload)
    if args.public_output is not None:
        _write_json(args.public_output, _public_payload(payload))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
