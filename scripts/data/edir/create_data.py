#!/usr/bin/env python3
"""Build local MTEB ``corpus``, ``queries``, and ``qrels`` datasets from EDIR.

The script downloads EDIR's three JSONL manifests and streams its split image
archive from Hugging Face. It extracts only images referenced by the manifests,
then materializes local MTEB datasets. It does not calculate statistics or
upload to the Hub.

Usage:
    python scripts/data/edir/create_data.py --work-dir data/edir
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi, snapshot_download
from PIL import Image as PILImage
from requests import Session

_SOURCE_REPO = "EDIR-BENCH/EDIR"
_SOURCE_REVISION = "2aed46dc8473b0068e978a98c61f4843b0a586a7"
_MANIFESTS = ("queries.jsonl", "corpus.jsonl", "instances.jsonl")
_ARCHIVE_PREFIX = "imgs_mc.tar.part"


@dataclass(frozen=True)
class SourceData:
    corpus: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    instances: list[dict[str, Any]]


class _RemoteArchiveParts(io.RawIOBase):
    """Expose ordered HTTP archive fragments as one read-only stream."""

    def __init__(self, urls: list[str]) -> None:
        self._urls: Iterator[str] = iter(urls)
        self._session = Session()
        self._response = None
        self._iterator = None
        self._buffer = b""
        self._closed = False

    def readable(self) -> bool:
        return True

    def _open_next_part(self) -> bool:
        try:
            url = next(self._urls)
        except StopIteration:
            return False
        self._response = self._session.get(url, stream=True, timeout=(30, 300))
        self._response.raise_for_status()
        self._iterator = self._response.iter_content(chunk_size=8 * 1024 * 1024)
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self._closed:
            return 0
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            if self._buffer:
                size = min(len(self._buffer), len(view) - total)
                view[total : total + size] = self._buffer[:size]
                self._buffer = self._buffer[size:]
                total += size
                continue
            if self._iterator is None and not self._open_next_part():
                break
            assert self._iterator is not None
            try:
                self._buffer = next(self._iterator)
            except StopIteration:
                assert self._response is not None
                self._response.close()
                self._response = None
                self._iterator = None
        return total

    def close(self) -> None:
        if not self._closed:
            if self._response is not None:
                self._response.close()
            self._session.close()
            self._closed = True
        super().close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _download_manifests(work_dir: Path) -> Path:
    return Path(
        snapshot_download(
            _SOURCE_REPO,
            repo_type="dataset",
            revision=_SOURCE_REVISION,
            allow_patterns=list(_MANIFESTS),
            local_dir=work_dir / "source",
        )
    )


def _load_and_validate_source(source_dir: Path) -> SourceData:
    source = SourceData(
        corpus=_read_jsonl(source_dir / "corpus.jsonl"),
        queries=_read_jsonl(source_dir / "queries.jsonl"),
        instances=_read_jsonl(source_dir / "instances.jsonl"),
    )
    corpus_ids = [str(row["id"]) for row in source.corpus]
    query_ids = [str(row["id"]) for row in source.queries]
    instance_ids = [str(row["qid"]) for row in source.instances]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("EDIR corpus contains duplicate IDs")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("EDIR queries contain duplicate IDs")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("EDIR instances contain duplicate query IDs")
    if set(query_ids) != set(instance_ids):
        raise ValueError("EDIR query and instance IDs do not match")

    corpus_id_set = set(corpus_ids)
    for row in source.corpus:
        if str(row.get("image")) != str(row["id"]):
            raise ValueError(f"Corpus image does not match ID: {row!r}")
    for row in source.queries:
        if not isinstance(row.get("image"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid query row: {row!r}")
    for row in source.instances:
        positives = row.get("pos")
        negatives = row.get("neg")
        if not isinstance(positives, list) or len(positives) != 1:
            raise ValueError(f"Expected one positive candidate: {row!r}")
        if not isinstance(negatives, list):
            raise ValueError(f"Invalid negative candidates: {row!r}")
        candidates = [str(candidate) for candidate in positives + negatives]
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"Duplicate candidates for query {row['qid']}")
        missing = set(candidates) - corpus_id_set
        if missing:
            raise ValueError(
                f"Candidates absent from corpus for {row['qid']}: {sorted(missing)}"
            )
    return source


def _required_media(source: SourceData) -> set[str]:
    return {
        *(str(row["image"]) for row in source.corpus),
        *(str(row["image"]) for row in source.queries),
    }


def _media_is_complete(media_dir: Path, required: set[str]) -> bool:
    if not media_dir.is_dir():
        return False
    if not required.issubset({path.name for path in media_dir.iterdir()}):
        return False
    for name in required:
        try:
            with PILImage.open(media_dir / name) as image:
                image.verify()
        except (OSError, SyntaxError):
            return False
    return True


def _archive_urls() -> list[str]:
    files = HfApi().list_repo_files(
        _SOURCE_REPO, repo_type="dataset", revision=_SOURCE_REVISION
    )
    names = sorted(name for name in files if name.startswith(_ARCHIVE_PREFIX))
    if not names:
        raise RuntimeError("No EDIR image archive fragments found")
    return [
        f"https://huggingface.co/datasets/{_SOURCE_REPO}/resolve/{_SOURCE_REVISION}/{name}"
        for name in names
    ]


def _extract_required_remote_media(
    urls: list[str], output_dir: Path, required: set[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    stream = _RemoteArchiveParts(urls)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name).name
                if name not in required:
                    continue
                destination = output_dir / name
                if destination.exists():
                    try:
                        with PILImage.open(destination) as image:
                            image.verify()
                    except (OSError, SyntaxError):
                        destination.unlink()
                    else:
                        extracted.add(name)
                        continue
                member_file = archive.extractfile(member)
                if member_file is None:
                    raise ValueError(f"Cannot extract EDIR archive member: {member.name}")
                with destination.open("wb") as target:
                    shutil.copyfileobj(member_file, target)
                extracted.add(name)
    finally:
        stream.close()
    missing = required - extracted
    if missing:
        raise FileNotFoundError(f"EDIR archive is missing {len(missing)} images")


def _build_datasets(source: SourceData, media_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    instances_by_qid = {str(row["qid"]): row for row in source.instances}
    corpus = Dataset.from_dict(
        {
            "id": [str(row["id"]) for row in source.corpus],
            "image": [str(media_dir / str(row["image"])) for row in source.corpus],
        }
    ).cast_column("image", Image())
    queries = Dataset.from_dict(
        {
            "id": [str(row["id"]) for row in source.queries],
            "image": [str(media_dir / str(row["image"])) for row in source.queries],
            "text": [str(row["text"]) for row in source.queries],
            "subcategory": [
                str(row["metadata"]["subcategory"]) for row in source.queries
            ],
        }
    ).cast_column("image", Image())
    qrels = Dataset.from_dict(
        {
            "query-id": [str(row["id"]) for row in source.queries],
            "corpus-id": [
                str(instances_by_qid[str(row["id"])]["pos"][0])
                for row in source.queries
            ],
            "score": [1] * len(source.queries),
        }
    )
    return corpus, queries, qrels


def _save_datasets(
    output_dir: Path, corpus: Dataset, queries: Dataset, qrels: Dataset
) -> None:
    DatasetDict({"test": corpus}).save_to_disk(str(output_dir / "corpus"))
    DatasetDict({"test": queries}).save_to_disk(str(output_dir / "queries"))
    DatasetDict({"test": qrels}).save_to_disk(str(output_dir / "qrels"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("data/edir"))
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    source = _load_and_validate_source(_download_manifests(work_dir))
    media_dir = work_dir / "media"
    required = _required_media(source)
    if _media_is_complete(media_dir, required):
        print(f"Using validated local media: {len(required)} images")
    else:
        _extract_required_remote_media(_archive_urls(), media_dir, required)
    corpus, queries, qrels = _build_datasets(source, media_dir)
    output_dir = work_dir / "mteb_export"
    _save_datasets(output_dir, corpus, queries, qrels)
    print(
        f"Saved corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)} "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
