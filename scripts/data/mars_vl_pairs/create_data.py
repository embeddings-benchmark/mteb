#!/usr/bin/env python3
"""Audit and freeze SUSTech/Mars-VL-Pairs with embedded image bytes.

The output is one ordered pair table. MTEB derives both text-to-image and
image-to-text retrieval tasks from this exact table so attrition cannot diverge
between directions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import statistics
import threading
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
from datasets import Dataset, DatasetDict, Image, load_dataset
from huggingface_hub import DatasetCard, HfApi, create_repo, get_token
from PIL import Image as PILImage
from PIL import ImageOps
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

SOURCE = "SUSTech/Mars-VL-Pairs"
SOURCE_REVISION = "1cac8885e481256d3752ad0d3a0f8f9681c5f206"
EXPECTED_PAIRS = 2_287
REQUIRED_COLUMNS = ("key", "image_url", "ori_caption", "refined_caption")
# Manual review of all dHash-distance <= 2 pairs found two resize-equivalent
# images with different captions. Retain the higher-resolution source member so
# every frozen query still has exactly one unambiguous positive.
PRACTICAL_DUPLICATE_GROUPS = ((460, 907), (1446, 2098))
EXCLUDED_PRACTICAL_DUPLICATE_INDICES = frozenset({907, 2098})
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
IMAGE_EXTENSIONS = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
_THREAD_LOCAL = threading.local()


@dataclass
class AuditRow:
    index: int
    key: str
    source_url: str
    ok: bool
    attempts: list[str]
    resolved_url: str | None = None
    status_code: int | None = None
    redirect_chain: list[dict[str, Any]] | None = None
    recovery_method: str = "direct"
    error: str | None = None
    local_path: str | None = None
    byte_sha256: str | None = None
    pixel_sha256: str | None = None
    dhash: str | None = None
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    mode: str | None = None
    file_size: int | None = None


def _session(retries: int) -> requests.Session:
    cache_key = f"session_{retries}"
    session = getattr(_THREAD_LOCAL, cache_key, None)
    if session is None:
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.75,
            status_forcelist=(408, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
        )
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        setattr(_THREAD_LOCAL, cache_key, session)
    return session


def _normalise_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _safe_stem(index: int, key: str) -> str:
    clean_key = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or "row"
    return f"{index:04d}-{clean_key}"


def _dhash(image: PILImage.Image) -> str:
    gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
    pixels = list(gray.tobytes())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def _validate_image(content: bytes) -> dict[str, Any]:
    with PILImage.open(BytesIO(content)) as check:
        check.verify()
    with PILImage.open(BytesIO(content)) as image:
        image.load()
        width, height = image.size
        image_format = (image.format or "").upper()
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions {width}x{height}")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixel_header = f"{rgb.width}x{rgb.height}:RGB:".encode()
        pixel_sha256 = hashlib.sha256(pixel_header + rgb.tobytes()).hexdigest()
        return {
            "width": width,
            "height": height,
            "format": image_format,
            "mode": image.mode,
            "pixel_sha256": pixel_sha256,
            "dhash": _dhash(image),
            "extension": IMAGE_EXTENSIONS.get(image_format, ".img"),
        }


def _fetch_bytes(
    url: str,
    *,
    retries: int,
    connect_timeout: float,
    read_timeout: float,
    max_bytes: int,
    verify_tls: bool = True,
) -> tuple[bytes, requests.Response]:
    parsed = urlparse(url)
    with warnings.catch_warnings():
        if not verify_tls:
            warnings.simplefilter("ignore", InsecureRequestWarning)
        response = _session(retries).get(
            url,
            allow_redirects=True,
            headers={"Referer": f"{parsed.scheme}://{parsed.netloc}/"},
            stream=True,
            timeout=(connect_timeout, read_timeout),
            verify=verify_tls,
        )
    response.raise_for_status()
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=1024 * 256):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("empty response")
    return b"".join(chunks), response


def _wayback_candidates(url: str, retries: int) -> list[str]:
    cdx_url = (
        "https://web.archive.org/cdx/search/cdx?"
        f"url={quote(url, safe='')}&output=json&fl=timestamp,original,statuscode,mimetype"
        "&filter=statuscode:200&collapse=digest&limit=10"
        "&sort=reverse"
    )
    try:
        response = _session(retries).get(cdx_url, timeout=(20, 60))
        response.raise_for_status()
        rows = response.json()
    except (requests.RequestException, ValueError):
        return []
    candidates = []
    if isinstance(rows, list) and len(rows) >= 2:
        for row in rows[1:]:
            if not isinstance(row, list) or len(row) < 2:
                continue
            timestamp, original = row[0], row[1]
            candidates.append(f"https://web.archive.org/web/{timestamp}id_/{original}")

    availability_url = (
        f"https://archive.org/wayback/available?url={quote(url, safe='')}"
    )
    try:
        response = _session(retries).get(availability_url, timeout=(20, 60))
        response.raise_for_status()
        closest = response.json().get("archived_snapshots", {}).get("closest", {})
        snapshot_url = closest.get("url") if closest.get("available") else None
        if snapshot_url:
            raw_url = re.sub(r"(/web/\d+)(?:[a-z_]+)?/", r"\1id_/", snapshot_url)
            candidates.append(raw_url)
    except (requests.RequestException, ValueError, AttributeError):
        pass
    return candidates


def _wikimedia_original(url: str) -> str | None:
    """Return the underlying MediaWiki file URL for a thumbnail URL."""
    parsed = urlparse(url)
    marker = "/wikipedia/commons/thumb/"
    if parsed.netloc.lower() != "upload.wikimedia.org" or marker not in parsed.path:
        return None
    prefix, remainder = parsed.path.split(marker, 1)
    parts = remainder.split("/")
    if len(parts) < 4:
        return None
    original_path = "/".join(parts[:-1])
    return parsed._replace(
        path=f"{prefix}/wikipedia/commons/{original_path}", query="", fragment=""
    ).geturl()


def _recovery_candidates(source_url: str, retries: int) -> list[tuple[str, str, bool]]:
    direct_variants = [source_url]
    unescaped = html.unescape(source_url)
    if unescaped != source_url:
        direct_variants.append(unescaped)

    candidates: list[tuple[str, str, bool]] = []
    for variant in direct_variants:
        candidates.append(
            (
                variant,
                "direct" if variant == source_url else "html-unescape",
                True,
            )
        )
        parsed = urlparse(variant)
        alternate_scheme = "https" if parsed.scheme == "http" else "http"
        candidates.append(
            (
                parsed._replace(scheme=alternate_scheme).geturl(),
                f"scheme-{alternate_scheme}",
                True,
            )
        )
        if parsed.scheme == "https":
            candidates.append((variant, "direct-tls-unverified", False))
    for variant in direct_variants:
        candidates.extend(
            (candidate, "wayback-exact-url", True)
            for candidate in _wayback_candidates(variant, retries)
        )
    wikimedia_original = _wikimedia_original(unescaped)
    if wikimedia_original:
        candidates.append((wikimedia_original, "wikimedia-original-file", True))
    return candidates


def _download_row(
    index: int,
    row: dict[str, Any],
    image_dir: Path,
    *,
    retries: int,
    connect_timeout: float,
    read_timeout: float,
    max_bytes: int,
    archive_recovery: bool,
) -> AuditRow:
    source_url = row["image_url"]
    result = AuditRow(
        index=index,
        key=str(row["key"]),
        source_url=source_url,
        ok=False,
        attempts=[],
    )
    candidates = (
        _recovery_candidates(source_url, retries)
        if archive_recovery
        else [(source_url, "direct", True)]
    )

    errors: list[str] = []
    attempted_candidates: set[tuple[str, bool]] = set()
    for candidate, recovery_method, verify_tls in candidates:
        candidate_key = (candidate, verify_tls)
        if candidate_key in attempted_candidates:
            continue
        attempted_candidates.add(candidate_key)
        attempt_label = candidate if verify_tls else f"{candidate} [TLS unverified]"
        result.attempts.append(attempt_label)
        try:
            content, response = _fetch_bytes(
                candidate,
                retries=retries,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_bytes=max_bytes,
                verify_tls=verify_tls,
            )
            image_info = _validate_image(content)
            extension = image_info.pop("extension")
            path = image_dir / f"{_safe_stem(index, result.key)}{extension}"
            path.write_bytes(content)
            result.ok = True
            result.resolved_url = response.url
            result.status_code = response.status_code
            result.redirect_chain = [
                {"status_code": item.status_code, "url": item.url}
                for item in response.history
            ]
            result.recovery_method = recovery_method
            result.local_path = str(path)
            result.byte_sha256 = hashlib.sha256(content).hexdigest()
            result.file_size = len(content)
            result.width = image_info["width"]
            result.height = image_info["height"]
            result.image_format = image_info["format"]
            result.mode = image_info["mode"]
            result.pixel_sha256 = image_info["pixel_sha256"]
            result.dhash = image_info["dhash"]
            return result
        except Exception as exc:  # errors are preserved in the audit manifest
            errors.append(f"{attempt_label}: {type(exc).__name__}: {exc}")
    result.error = " | ".join(errors)
    return result


def _duplicate_groups(values: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def _near_dhash_pairs(
    rows: list[AuditRow], max_distance: int = 2
) -> list[dict[str, Any]]:
    values = [(row.index, int(row.dhash, 16)) for row in rows if row.dhash]
    pairs = []
    for offset, (left_index, left_hash) in enumerate(values):
        for right_index, right_hash in values[offset + 1 :]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_distance:
                pairs.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "distance": distance,
                    }
                )
    return pairs


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _load_reusable_audit(
    work_dir: Path, source_rows: list[dict[str, Any]]
) -> list[AuditRow | None]:
    """Reuse validated successes so a recovery pass only touches failed URLs."""
    manifest_path = work_dir / "audit_rows.jsonl"
    reusable: list[AuditRow | None] = [None] * len(source_rows)
    if not manifest_path.exists():
        return reusable
    manifest = [
        AuditRow(**json.loads(line))
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]
    for row in manifest:
        if not row.ok or row.index >= len(source_rows) or row.local_path is None:
            continue
        source_row = source_rows[row.index]
        if (
            row.key != str(source_row["key"])
            or row.source_url != source_row["image_url"]
        ):
            continue
        path = Path(row.local_path)
        if not path.is_file() or row.byte_sha256 is None:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.byte_sha256:
            continue
        reusable[row.index] = row
    print(f"Reusing {sum(row is not None for row in reusable)} validated downloads")
    return reusable


def _build_summary(
    source_rows: list[dict[str, Any]], audit_rows: list[AuditRow]
) -> dict[str, Any]:
    recovered = [row for row in audit_rows if row.ok]
    widths = [row.width for row in recovered if row.width is not None]
    heights = [row.height for row in recovered if row.height is not None]
    areas = [width * height for width, height in zip(widths, heights)]
    source_domains = Counter(
        urlparse(row["image_url"]).netloc.lower().removeprefix("www.")
        for row in source_rows
    )
    byte_groups = _duplicate_groups([row.byte_sha256 or "" for row in recovered])
    pixel_groups = _duplicate_groups([row.pixel_sha256 or "" for row in recovered])
    return {
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "expected_rows": EXPECTED_PAIRS,
        "source_rows": len(source_rows),
        "recovered_rows": len(recovered),
        "missing_rows": len(audit_rows) - len(recovered),
        "frozen_rows": len(recovered)
        - sum(audit_rows[index].ok for index in EXCLUDED_PRACTICAL_DUPLICATE_INDICES),
        "practical_duplicate_groups": [
            list(group) for group in PRACTICAL_DUPLICATE_GROUPS
        ],
        "excluded_practical_duplicate_indices": sorted(
            EXCLUDED_PRACTICAL_DUPLICATE_INDICES
        ),
        "failed_indices": [row.index for row in audit_rows if not row.ok],
        "source_url_duplicates": _duplicate_groups(
            [row["image_url"] for row in source_rows]
        ),
        "key_duplicates": _duplicate_groups([str(row["key"]) for row in source_rows]),
        "original_caption_duplicates": _duplicate_groups(
            [_normalise_text(row["ori_caption"]) for row in source_rows]
        ),
        "refined_caption_duplicates": _duplicate_groups(
            [_normalise_text(row["refined_caption"]) for row in source_rows]
        ),
        "byte_duplicate_groups": byte_groups,
        "pixel_duplicate_groups": pixel_groups,
        "near_dhash_pairs_distance_le_2": _near_dhash_pairs(recovered),
        "redirected_rows": sum(bool(row.redirect_chain) for row in recovered),
        "recovery_methods": dict(Counter(row.recovery_method for row in recovered)),
        "formats": dict(Counter(row.image_format for row in recovered)),
        "modes": dict(Counter(row.mode for row in recovered)),
        "source_schemes": dict(
            Counter(urlparse(row["image_url"]).scheme for row in source_rows)
        ),
        "source_domain_count": len(source_domains),
        "source_domains": dict(source_domains.most_common()),
        "total_image_bytes": sum(row.file_size or 0 for row in recovered),
        "dimensions": {
            "min_width": min(widths, default=None),
            "max_width": max(widths, default=None),
            "median_width": statistics.median(widths) if widths else None,
            "min_height": min(heights, default=None),
            "max_height": max(heights, default=None),
            "median_height": statistics.median(heights) if heights else None,
            "min_pixels": min(areas, default=None),
            "max_pixels": max(areas, default=None),
            "median_pixels": statistics.median(areas) if areas else None,
        },
    }


def _make_dataset(
    source_rows: list[dict[str, Any]], audit_rows: list[AuditRow]
) -> DatasetDict:
    audit_by_index = {row.index: row for row in audit_rows if row.ok}
    records = []
    for index, source_row in enumerate(source_rows):
        if index in EXCLUDED_PRACTICAL_DUPLICATE_INDICES:
            continue
        audit = audit_by_index.get(index)
        if audit is None:
            continue
        records.append(
            {
                "key": str(source_row["key"]),
                "image": audit.local_path,
                "ori_caption": source_row["ori_caption"],
                "refined_caption": source_row["refined_caption"],
                "source_url": source_row["image_url"],
                "resolved_url": audit.resolved_url or "",
                "source_domain": urlparse(source_row["image_url"])
                .netloc.lower()
                .removeprefix("www."),
                "recovery_method": audit.recovery_method,
                "content_sha256": audit.byte_sha256,
                "pixel_sha256": audit.pixel_sha256,
                "width": audit.width,
                "height": audit.height,
                "image_format": audit.image_format,
                "file_size": audit.file_size,
            }
        )
    dataset = Dataset.from_list(records).cast_column("image", Image())
    return DatasetDict({"test": dataset})


def _card_appendix(summary: dict[str, Any]) -> str:
    return f"""

# Mars-VL-Pairs MTEB frozen media release

This is a reproducibility-focused mirror of Task 1 from MarsRetrieval. It embeds
the image bytes used by the benchmark instead of downloading mutable web URLs
at MTEB evaluation time. The original row order, key, original caption, refined
caption, source URL, resolved URL, hashes, dimensions, and recovery method are
preserved. MTEB uses `refined_caption`, matching the paper's main evaluation.

## Construction and integrity

- Source: `{SOURCE}` at `{SOURCE_REVISION}`
- Expected source pairs: {summary["expected_rows"]}
- Recovered media rows: {summary["recovered_rows"]}
- Missing pairs: {summary["missing_rows"]}
- Evaluation pairs after duplicate review: {summary["frozen_rows"]}
- Excluded resize-equivalent rows: {summary["excluded_practical_duplicate_indices"]}
- Redirected downloads: {summary["redirected_rows"]}
- Exact byte duplicate groups: {len(summary["byte_duplicate_groups"])}
- Decoded-pixel duplicate groups: {len(summary["pixel_duplicate_groups"])}
- Source domains: {summary["source_domain_count"]}

Both image-to-text and text-to-image tasks use this same frozen pair table.
Every row has exactly one paired refined caption and image.

## License and provenance

The source dataset declares CC-BY-4.0. The MarsRetrieval paper states that Task
1 candidates were selected from DataComp-1B and Relation-2B, and the retained
images resolve to many external web domains. The dataset-level license metadata
may not supersede rights or attribution requirements attached to each original
web image. The `source_url` and `source_domain` fields are retained so users can
inspect provenance and applicable source terms.

## Citation

```bibtex
@article{{wang2026marsretrieval,
  title={{MarsRetrieval: Benchmarking Vision-Language Models for Planetary-Scale Geospatial Retrieval on Mars}},
  author={{Wang, Shuoyuan and Wang, Yiran and Wei, Hongxin}},
  journal={{arXiv preprint arXiv:2602.13961}},
  year={{2026}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--repo-id", default="Cerru02/Mars-VL-Pairs-MTEB")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/mars_vl_pairs_mteb")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=20)
    parser.add_argument("--read-timeout", type=float, default=60)
    parser.add_argument("--max-image-mb", type=int, default=128)
    parser.add_argument("--archive-recovery", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-duplicate-images", action="store_true")
    parser.add_argument("--redownload-all", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.source_revision != SOURCE_REVISION:
        raise SystemExit(
            f"Refusing unreviewed source revision {args.source_revision}; "
            f"expected {SOURCE_REVISION}"
        )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    source = load_dataset(
        SOURCE,
        revision=args.source_revision,
        split="train",
    )
    missing_columns = set(REQUIRED_COLUMNS) - set(source.column_names)
    if missing_columns:
        raise SystemExit(f"Missing source columns: {sorted(missing_columns)}")
    if len(source) != EXPECTED_PAIRS:
        raise SystemExit(f"Expected {EXPECTED_PAIRS} rows, found {len(source)}")
    source_rows = source.select_columns(REQUIRED_COLUMNS).to_list()
    for index, row in enumerate(source_rows):
        for column in REQUIRED_COLUMNS:
            if not isinstance(row[column], str) or not row[column].strip():
                raise SystemExit(f"Empty {column} at source row {index}")

    futures = {}
    audit_rows = (
        [None] * len(source_rows)
        if args.redownload_all
        else _load_reusable_audit(args.work_dir, source_rows)
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(source_rows):
            if audit_rows[index] is not None:
                continue
            future = executor.submit(
                _download_row,
                index,
                row,
                image_dir,
                retries=args.retries,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                max_bytes=args.max_image_mb * 1024 * 1024,
                archive_recovery=args.archive_recovery,
            )
            futures[future] = index
        with tqdm(total=len(futures), desc="download and validate") as progress:
            for future in as_completed(futures):
                index = futures[future]
                audit_rows[index] = future.result()
                progress.update(1)

    completed_rows = [row for row in audit_rows if row is not None]
    if len(completed_rows) != len(source_rows):
        raise RuntimeError("Audit did not return every source row")
    summary = _build_summary(source_rows, completed_rows)
    _write_json(args.work_dir / "audit_summary.json", summary)
    _write_jsonl(
        args.work_dir / "audit_rows.jsonl",
        [asdict(row) for row in completed_rows],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["missing_rows"] and not args.allow_missing:
        raise SystemExit(
            "Some images are missing. Review audit_rows.jsonl, then retry with "
            "--archive-recovery or explicitly accept deterministic attrition with "
            "--allow-missing."
        )
    if (
        summary["byte_duplicate_groups"] or summary["pixel_duplicate_groups"]
    ) and not args.allow_duplicate_images:
        raise SystemExit(
            "Duplicate image content would make one-positive qrels ambiguous. "
            "Review audit_summary.json before using --allow-duplicate-images."
        )
    if summary["refined_caption_duplicates"]:
        raise SystemExit(
            "Normalized refined-caption duplicates would make text queries ambiguous."
        )

    dataset = _make_dataset(source_rows, completed_rows)
    export_dir = args.work_dir / "dataset"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    dataset.save_to_disk(export_dir)
    print(f"Saved {len(dataset['test'])} frozen pairs to {export_dir}")

    if not args.push:
        return
    token = get_token()
    if not token:
        raise SystemExit("No authenticated Hugging Face token found")
    api = HfApi(token=token)
    identity = api.whoami()["name"]
    namespace = args.repo_id.split("/", 1)[0]
    if identity.casefold() != namespace.casefold():
        raise SystemExit(
            f"Authenticated as {identity}, refusing to push to namespace {namespace}"
        )
    create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
    dataset.push_to_hub(args.repo_id, token=token, max_shard_size="1GB")
    card = DatasetCard.load(args.repo_id, token=token)
    DatasetCard(str(card).rstrip() + _card_appendix(summary)).push_to_hub(
        args.repo_id, token=token
    )
    revision = api.dataset_info(args.repo_id).sha
    print(f"Pushed https://huggingface.co/datasets/{args.repo_id}")
    print(f"Immutable revision: {revision}")


if __name__ == "__main__":
    main()
