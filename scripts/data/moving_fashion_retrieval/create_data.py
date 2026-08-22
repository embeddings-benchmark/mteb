#!/usr/bin/env python3
"""Build the native MovingFashion video-to-image retrieval benchmark for MTEB.

The official archive contains train and test annotations plus the corresponding
Instagram videos and Net-A-Porter shop images. MTEB uses every available query
from the official test split: videos are queries, shop images form the corpus,
and every usable annotated video/image association is a binary relevance
judgment. Repeated media paths are collapsed by path, preserving the source's
genuine multi-positive cases.

The published Hugging Face dataset retains the source benchmark's video-to-shop
image orientation.

Examples:
  # Download or resume the official source archive.
  ./scripts/data/moving_fashion_retrieval/download_source.sh movingfashion.zip

  # Audit annotations extracted from the archive (no media required).
  uv run python scripts/data/moving_fashion_retrieval/create_data.py \
      --annotation-dir /path/to/annotations

  # Audit the complete archive, extract only test media, validate it, and save
  # the standard MTEB retrieval configs locally.
  uv run python scripts/data/moving_fashion_retrieval/create_data.py \
      --archive /path/to/movingfashion.zip --save-to-disk

  # Publish using the currently authenticated Hugging Face account.
  uv run python scripts/data/moving_fashion_retrieval/create_data.py \
      --archive /path/to/movingfashion.zip --push \
      --repo-id pranitchawla/MovingFashion
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import Dataset, DatasetDict, Image, Value, Video
from huggingface_hub import HfApi, create_repo, get_token
from PIL import Image as PILImage

_PROJECT_URL = "https://humaticslab.github.io/retrieval/movingfashion"
_PAPER_URL = "https://arxiv.org/abs/2110.02627"
_SOURCE_REPO_URL = "https://github.com/HumaticsLAB/SEAM-Match-RCNN"
_SOURCE_REPO_REVISION = "4ca15d147ce87f0385c0c9779eac49e55c727ec8"
_SOURCE_DOWNLOAD_URL = "https://bit.ly/4bTZGeS"
_LICENSE = "cc-by-nc-sa-4.0"
_EXPECTED_ARCHIVE_SHA256 = (
    "20ae89a67a58d3dfc2304c2533d5a5c684eb6888aec9a7d7d3eda22b0a81f5f4"
)


@dataclass(frozen=True)
class ExpectedSplit:
    products: int
    associations: int
    unique_videos: int
    unique_images: int
    source_hard: int
    source_regular: int
    products_with_multiple_videos: int
    max_videos_per_product: int


_EXPECTED_SPLITS = {
    "train": ExpectedSplit(
        products=12648,
        associations=13703,
        unique_videos=13526,
        unique_images=12630,
        source_hard=3530,
        source_regular=9118,
        products_with_multiple_videos=573,
        max_videos_per_product=24,
    ),
    "test": ExpectedSplit(
        products=1342,
        associations=1342,
        unique_videos=1329,
        unique_images=1341,
        source_hard=328,
        source_regular=1014,
        products_with_multiple_videos=0,
        max_videos_per_product=1,
    ),
}
_EXPECTED_TEST_UNIQUE_PAIRS = 1342
_EXPECTED_TEST_DUPLICATE_VIDEO_GROUPS = 13
_EXPECTED_TEST_DUPLICATE_IMAGE_GROUPS = 1
_EXPECTED_TOTAL_UNIQUE_VIDEOS = 14855
_EXPECTED_MISSING_MEDIA = {
    "train": {
        "videos/0096903c8314b3870a7af2a425953536.mp4",
        "videos/0476509b035f59bfc009827ac4d8bafc.mp4",
        "videos/0580eb0b3128457511e09c4503391e64.mp4",
        "videos/1242272_detail.mp4",
        "videos/154532233c6ad4de20f01bd51b0fc1d0.mp4",
        "videos/267e49a2051bbaedfe76d3ee2a20cd58.mp4",
        "videos/3360fdc695014ce440aff4cb97afadbb.mp4",
        "videos/47780e36140c63e91131555823fef909.mp4",
        "videos/5330d0e23a7327d55685380893109692.mp4",
        "videos/57893f18394ef329b66e35c345608694.mp4",
        "videos/583c1c541b19369daa6ab457c9239455.mp4",
        "videos/5f2fc22016a5e6935543dba09744032e.mp4",
        "videos/784d4f0f64ddca7ff8d7c30b4d356e07.mp4",
        "videos/7a8f7e95a14ed9d5f61875592ccc052b.mp4",
        "videos/7ead8a1d0646890d2d509c91ce2e4872.mp4",
        "videos/97210917e5c1423fb977e1ad54a3434b.mp4",
        "videos/b2d1b95537f74557b78636c417a409ce.mp4",
        "videos/b959460caa57d3a0144f097756485106.mp4",
        "videos/c8ec36fdca477f3828313f91cbf2d370.mp4",
        "videos/e1fa86dc926d29771ee89d8bc811487e.mp4",
        "videos/e5e722166385f10fd462af377e3be7b5.mp4",
        "videos/f386b23468d1ffda850fef2f8aa438cb.mp4",
    },
    "test": {"videos/bd6186ecf66d1a5c2d65ff3c891c7e44.mp4"},
}


@dataclass(frozen=True, order=True)
class Association:
    product_id: str
    video_path: str
    image_path: str
    source: int


def _validate_media_path(path: str, prefix: str) -> str:
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.parts[0] != prefix
    ):
        raise RuntimeError(f"Unexpected {prefix} path in annotations: {path!r}")
    normalized = pure_path.as_posix()
    if normalized != path:
        raise RuntimeError(f"Non-canonical media path in annotations: {path!r}")
    return normalized


def _parse_annotations(
    raw: Any, split: str
) -> tuple[list[Association], Counter[int], Counter[str]]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{split}.json must contain a JSON object")

    associations: list[Association] = []
    source_counts: Counter[int] = Counter()
    videos_per_product: Counter[str] = Counter()
    for product_id, record in raw.items():
        if not isinstance(product_id, str) or not product_id:
            raise RuntimeError(f"Invalid product ID in {split}.json: {product_id!r}")
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid record for product {product_id}")

        image_path = record.get("img_path")
        video_paths = record.get("video_paths")
        source = record.get("source")
        if not isinstance(image_path, str):
            raise RuntimeError(f"Invalid image path for product {product_id}")
        image_path = _validate_media_path(image_path, "imgs")
        if (
            not isinstance(video_paths, list)
            or not video_paths
            or not all(isinstance(path, str) for path in video_paths)
        ):
            raise RuntimeError(f"Invalid video paths for product {product_id}")
        if source not in {0, 1}:
            raise RuntimeError(
                f"Invalid source value for product {product_id}: {source!r}"
            )
        if len(video_paths) != len(set(video_paths)):
            raise RuntimeError(f"Repeated video path within product {product_id}")

        source_counts[source] += 1
        videos_per_product[product_id] = len(video_paths)
        associations.extend(
            Association(
                product_id=product_id,
                video_path=_validate_media_path(video_path, "videos"),
                image_path=image_path,
                source=source,
            )
            for video_path in video_paths
        )

    if len(associations) != len(set(associations)):
        raise RuntimeError(f"{split}.json contains duplicate association rows")
    return associations, source_counts, videos_per_product


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_annotations_from_directory(
    annotation_dir: Path,
) -> dict[str, tuple[list[Association], Counter[int], Counter[str]]]:
    output = {}
    for split in _EXPECTED_SPLITS:
        path = annotation_dir / f"{split}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing annotation file: {path}")
        output[split] = _parse_annotations(_load_json(path), split)
    return output


def _zip_member_map(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in source archive: {info.filename!r}")
        normalized = path.as_posix().removeprefix("./")
        if normalized in members:
            raise RuntimeError(f"Duplicate path in source archive: {normalized}")
        members[normalized] = info
    return members


def _load_annotations_from_archive(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]
) -> dict[str, tuple[list[Association], Counter[int], Counter[str]]]:
    output = {}
    for split in _EXPECTED_SPLITS:
        filename = f"{split}.json"
        if filename not in members:
            raise RuntimeError(f"Missing {filename} in source archive")
        with archive.open(members[filename]) as handle:
            output[split] = _parse_annotations(json.load(handle), split)
    return output


def _split_summary(
    associations: list[Association],
    source_counts: Counter[int],
    videos_per_product: Counter[str],
) -> dict[str, Any]:
    videos = {row.video_path for row in associations}
    images = {row.image_path for row in associations}
    return {
        "products": len(videos_per_product),
        "associations": len(associations),
        "unique_videos": len(videos),
        "unique_images": len(images),
        "source_hard": source_counts[0],
        "source_regular": source_counts[1],
        "products_with_multiple_videos": sum(
            count > 1 for count in videos_per_product.values()
        ),
        "max_videos_per_product": max(videos_per_product.values(), default=0),
    }


def _audit_annotations(
    annotations: dict[str, tuple[list[Association], Counter[int], Counter[str]]],
    *,
    allow_source_changes: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_repository": _SOURCE_REPO_URL,
        "source_repository_revision": _SOURCE_REPO_REVISION,
        "source_download": _SOURCE_DOWNLOAD_URL,
        "license": _LICENSE,
        "splits": {},
    }
    for split, expected in _EXPECTED_SPLITS.items():
        actual = _split_summary(*annotations[split])
        summary["splits"][split] = actual
        if not allow_source_changes and actual != asdict(expected):
            raise RuntimeError(
                f"Unexpected {split} annotation statistics. Expected "
                f"{asdict(expected)}, found {actual}. Review the source before rebuilding."
            )

    train_associations = annotations["train"][0]
    test_associations = annotations["test"][0]
    train_products = {row.product_id for row in train_associations}
    train_videos = {row.video_path for row in train_associations}
    train_images = {row.image_path for row in train_associations}
    test_products = {row.product_id for row in test_associations}
    test_videos = {row.video_path for row in test_associations}
    test_images = {row.image_path for row in test_associations}
    overlap = {
        "products": sorted(train_products & test_products),
        "videos": sorted(train_videos & test_videos),
        "images": sorted(train_images & test_images),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Train/test leakage detected: {overlap}")

    unique_pairs = {(row.video_path, row.image_path) for row in test_associations}
    video_to_images: dict[str, set[str]] = defaultdict(set)
    image_to_videos: dict[str, set[str]] = defaultdict(set)
    for video_path, image_path in unique_pairs:
        video_to_images[video_path].add(image_path)
        image_to_videos[image_path].add(video_path)
    duplicate_video_groups = {
        video: sorted(images)
        for video, images in video_to_images.items()
        if len(images) > 1
    }
    duplicate_image_groups = {
        image: sorted(videos)
        for image, videos in image_to_videos.items()
        if len(videos) > 1
    }
    retrieval = {
        "v2i_queries": len(video_to_images),
        "v2i_corpus": len(image_to_videos),
        "v2i_qrels": len(unique_pairs),
        "i2v_queries": len(image_to_videos),
        "i2v_corpus": len(video_to_images),
        "i2v_qrels": len(unique_pairs),
        "video_queries_with_multiple_positives": len(duplicate_video_groups),
        "image_queries_with_multiple_positives": len(duplicate_image_groups),
        "duplicate_video_groups": duplicate_video_groups,
        "duplicate_image_groups": duplicate_image_groups,
    }
    summary["annotated_retrieval"] = retrieval
    summary["train_test_overlap"] = overlap
    summary["total_unique_videos"] = len(train_videos | test_videos)

    expected_retrieval = {
        "v2i_queries": _EXPECTED_SPLITS["test"].unique_videos,
        "v2i_corpus": _EXPECTED_SPLITS["test"].unique_images,
        "v2i_qrels": _EXPECTED_TEST_UNIQUE_PAIRS,
        "i2v_queries": _EXPECTED_SPLITS["test"].unique_images,
        "i2v_corpus": _EXPECTED_SPLITS["test"].unique_videos,
        "i2v_qrels": _EXPECTED_TEST_UNIQUE_PAIRS,
        "video_queries_with_multiple_positives": (
            _EXPECTED_TEST_DUPLICATE_VIDEO_GROUPS
        ),
        "image_queries_with_multiple_positives": (
            _EXPECTED_TEST_DUPLICATE_IMAGE_GROUPS
        ),
    }
    actual_retrieval = {key: retrieval[key] for key in expected_retrieval}
    if not allow_source_changes and actual_retrieval != expected_retrieval:
        raise RuntimeError(
            "Unexpected test retrieval structure. Expected "
            f"{expected_retrieval}, found {actual_retrieval}."
        )
    if (
        not allow_source_changes
        and summary["total_unique_videos"] != _EXPECTED_TOTAL_UNIQUE_VIDEOS
    ):
        raise RuntimeError(
            f"Expected {_EXPECTED_TOTAL_UNIQUE_VIDEOS} unique videos, found "
            f"{summary['total_unique_videos']}"
        )
    return summary


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _validate_archive_paths(
    members: dict[str, zipfile.ZipInfo],
    annotations: dict[str, tuple[Any, ...]],
    *,
    allow_source_changes: bool,
) -> dict[str, Any]:
    references_by_split = {
        split: {
            path
            for row in split_annotations[0]
            for path in (row.video_path, row.image_path)
        }
        for split, split_annotations in annotations.items()
    }
    referenced = set().union(*references_by_split.values())
    missing_by_split = {
        split: sorted(path for path in references if path not in members)
        for split, references in references_by_split.items()
    }
    actual_missing = {split: set(paths) for split, paths in missing_by_split.items()}
    if not allow_source_changes and actual_missing != _EXPECTED_MISSING_MEDIA:
        raise RuntimeError(
            "Unexpected source archive omissions. Expected "
            f"{_EXPECTED_MISSING_MEDIA}, found {actual_missing}."
        )
    archive_media = {
        path
        for path, info in members.items()
        if not info.is_dir() and path.startswith(("videos/", "imgs/"))
    }
    return {
        "referenced_media": len(referenced),
        "archive_media": len(archive_media),
        "unreferenced_archive_media": len(archive_media - referenced),
        "missing_referenced_media": missing_by_split,
        "missing_referenced_media_count": sum(map(len, missing_by_split.values())),
    }


def _extract_member(
    archive_path: Path,
    info: zipfile.ZipInfo,
    destination_root: Path,
) -> Path:
    destination = destination_root / PurePosixPath(info.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == info.file_size
        and _crc32(destination) == info.CRC
    ):
        return destination

    partial = destination.with_name(destination.name + ".part")
    with zipfile.ZipFile(archive_path) as archive, archive.open(info) as source:
        with partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if partial.stat().st_size != info.file_size or _crc32(partial) != info.CRC:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Failed CRC/size validation for {info.filename}")
    partial.replace(destination)
    return destination


def _extract_test_media(
    archive_path: Path,
    members: dict[str, zipfile.ZipInfo],
    test_associations: list[Association],
    media_dir: Path,
    *,
    workers: int,
) -> tuple[list[str], list[str]]:
    videos = sorted(
        {row.video_path for row in test_associations if row.video_path in members}
    )
    images = sorted(
        {row.image_path for row in test_associations if row.image_path in members}
    )
    paths = [*videos, *images]

    def extract(path: str) -> Path:
        return _extract_member(archive_path, members[path], media_dir)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(extract, paths))
    return videos, images


def _image_is_decodable(path: Path) -> bool:
    try:
        with PILImage.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _video_is_decodable(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate MovingFashion videos")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return False
    return bool(
        streams
        and streams[0].get("codec_name")
        and streams[0].get("width", 0) > 0
        and streams[0].get("height", 0) > 0
    )


def _validate_test_media(
    media_dir: Path,
    videos: list[str],
    images: list[str],
    *,
    workers: int,
) -> None:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        image_results = pool.map(
            lambda path: _image_is_decodable(media_dir / path), images
        )
        bad_images = [
            path for path, valid in zip(images, image_results, strict=True) if not valid
        ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        video_results = pool.map(
            lambda path: _video_is_decodable(media_dir / path), videos
        )
        bad_videos = [
            path for path, valid in zip(videos, video_results, strict=True) if not valid
        ]
    if bad_images or bad_videos:
        raise RuntimeError(
            f"Media validation failed: bad_images={bad_images}, bad_videos={bad_videos}"
        )


def _build_datasets(
    test_associations: list[Association],
    media_dir: Path,
    videos: list[str],
    images: list[str],
) -> tuple[Dataset, Dataset, Dataset]:
    video_ids = set(videos)
    image_ids = set(images)
    pairs = sorted(
        {
            (row.video_path, row.image_path)
            for row in test_associations
            if row.video_path in video_ids and row.image_path in image_ids
        }
    )
    video_ids = sorted(video_ids)
    image_ids = sorted(image_ids)
    queries = Dataset.from_dict(
        {
            "id": video_ids,
            "video": [str(media_dir / video_id) for video_id in video_ids],
        }
    ).cast_column("video", Video())
    corpus = Dataset.from_dict(
        {
            "id": image_ids,
            "image": [str(media_dir / image_id) for image_id in image_ids],
        }
    ).cast_column("image", Image())
    qrels = Dataset.from_dict(
        {
            "query-id": [video_path for video_path, _ in pairs],
            "corpus-id": [image_path for _, image_path in pairs],
            "score": [1] * len(pairs),
        }
    ).cast_column("score", Value("int32"))
    return corpus, queries, qrels


def _published_retrieval_summary(
    test_associations: list[Association], videos: list[str], images: list[str]
) -> dict[str, Any]:
    video_ids = set(videos)
    image_ids = set(images)
    pairs = {
        (row.video_path, row.image_path)
        for row in test_associations
        if row.video_path in video_ids and row.image_path in image_ids
    }
    images_with_qrels = {image_path for _, image_path in pairs}
    return {
        "v2i_queries": len(video_ids),
        "v2i_corpus": len(image_ids),
        "v2i_qrels": len(pairs),
        "corpus_images_without_qrels": sorted(image_ids - images_with_qrels),
    }


def _dataset_card(summary: dict[str, Any]) -> str:
    retrieval = summary["published_retrieval"]
    archive_sha256 = summary.get("archive_sha256", "not recorded")
    return f"""---
license: cc-by-nc-sa-4.0
pretty_name: MovingFashion Retrieval
tags:
- mteb
- moeb
- video-to-image
- cross-modal-retrieval
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
---

# MovingFashion Retrieval

Frozen MTEB/MOEB representation of the official MovingFashion test split for
video-to-shop-image fashion retrieval.

## Construction

The dataset is derived from the [official MovingFashion release]({_PROJECT_URL})
and its `test.json` associations. The MTEB construction script audits both
official splits against the source code at revision
[`{_SOURCE_REPO_REVISION}`]({_SOURCE_REPO_URL}/tree/{_SOURCE_REPO_REVISION}),
checks that train and test share no product IDs or media paths, verifies every
annotation reference against the archive, pins the known source omissions, and
decodes all published test media. The source archive SHA-256 for this build is
`{archive_sha256}`.

Media paths are used as IDs. Repeated paths are collapsed without discarding
associations, so the source's multi-positive relevance structure is preserved.
The Hub configs use the standard MTEB representation: `queries` contains video
queries, `corpus` contains shop images, and `qrels` contains binary relevance
judgments.

The official archive omits 22 train videos and one annotated test video. The
missing test query and its unusable qrel are excluded; its available shop image
remains in the corpus as a distractor. The construction script pins and reports
all 23 source omissions rather than silently dropping them.

## Evaluation contents

- Video-to-image: {retrieval["v2i_queries"]} queries,
  {retrieval["v2i_corpus"]} corpus images, and {retrieval["v2i_qrels"]} qrels.
- Corpus images without a qrel: {len(retrieval["corpus_images_without_qrels"])}.
- Source difficulty labels: `0` is hard and `1` is regular. They are audited
  during construction but are not used to filter the benchmark.

## Source protocol and baselines

The source benchmark evaluates video-to-shop retrieval with top-k accuracy. The
paper reports SEAM Match-RCNN top-1/5/10/20 accuracy of .49/.80/.89/.94 overall,
.55/.86/.94/.97 on the regular subset, and .30/.62/.76/.87 on the hard subset.
Those numbers use the original task-specific detector and are context rather
than directly comparable guarantees for generic embedding models.

## License, provenance, and limitations

The official repository labels the work CC BY-NC-SA 4.0 and says the dataset is
available for academic purposes. This derived release therefore retains
CC BY-NC-SA 4.0, attribution, non-commercial, and share-alike requirements.
Source videos originated on Instagram and shop images on Net-A-Porter; underlying
media, privacy, publicity, trademark, and platform rights may remain with their
respective owners. The paper states that faces were blurred. Users remain
responsible for determining whether their use complies with the license,
source-platform terms, and applicable law.

## Citation

```bibtex
@misc{{godi2021movingfashion,
  title = {{MovingFashion: a Benchmark for the Video-to-Shop Challenge}},
  author = {{Godi, Marco and Joppi, Christian and Skenderi, Geri and Cristani, Marco}},
  year = {{2021}},
  eprint = {{2110.02627}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.CV}},
}}
```

See the [paper]({_PAPER_URL}) and [official code]({_SOURCE_REPO_URL}).
"""


def _save_datasets(
    work_dir: Path, corpus: Dataset, queries: Dataset, qrels: Dataset
) -> None:
    export_dir = work_dir / "mteb_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"test": corpus}).save_to_disk(export_dir / "corpus")
    DatasetDict({"test": queries}).save_to_disk(export_dir / "queries")
    DatasetDict({"test": qrels}).save_to_disk(export_dir / "qrels")
    print(f"Wrote local dataset to {export_dir}")


def _publish(
    repo_id: str,
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
    summary: dict[str, Any],
    work_dir: Path,
) -> str:
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found; run `hf auth login` before --push"
        )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=_dataset_card(summary).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add MovingFashion dataset card",
    )
    DatasetDict({"test": corpus}).push_to_hub(
        repo_id,
        "corpus",
        token=token,
        max_shard_size="500MB",
        commit_message="Add MovingFashion image corpus",
    )
    DatasetDict({"test": queries}).push_to_hub(
        repo_id,
        "queries",
        token=token,
        max_shard_size="500MB",
        commit_message="Add MovingFashion video queries",
    )
    DatasetDict({"test": qrels}).push_to_hub(
        repo_id,
        "qrels",
        token=token,
        commit_message="Add MovingFashion relevance judgments",
    )
    revision = api.dataset_info(repo_id).sha
    (work_dir / "hub_revision.txt").write_text(f"{revision}\n", encoding="utf-8")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--archive", type=Path, help="Path to the official movingfashion.zip"
    )
    source.add_argument(
        "--annotation-dir",
        type=Path,
        help="Directory containing extracted train.json and test.json for audit only",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/moving_fashion_retrieval")
    )
    parser.add_argument("--repo-id", default="pranitchawla/MovingFashion")
    parser.add_argument("--save-to-disk", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--allow-source-changes",
        action="store_true",
        help="Report rather than reject changes to the pinned annotation statistics",
    )
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--verify-workers", type=int, default=8)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    archive: zipfile.ZipFile | None = None
    members: dict[str, zipfile.ZipInfo] | None = None
    if args.annotation_dir is not None:
        if args.save_to_disk or args.push:
            raise RuntimeError(
                "--save-to-disk and --push require --archive so media can be included"
            )
        annotations = _load_annotations_from_directory(args.annotation_dir.resolve())
    else:
        archive_path = args.archive.resolve()
        if not archive_path.is_file():
            raise RuntimeError(f"Archive not found: {archive_path}")
        archive = zipfile.ZipFile(archive_path)
        members = _zip_member_map(archive)
        annotations = _load_annotations_from_archive(archive, members)

    summary = _audit_annotations(
        annotations, allow_source_changes=args.allow_source_changes
    )
    corpus: Dataset | None = None
    queries: Dataset | None = None
    qrels: Dataset | None = None
    if archive is not None and members is not None:
        summary["archive"] = _validate_archive_paths(
            members, annotations, allow_source_changes=args.allow_source_changes
        )
        archive.close()
        summary["archive_sha256"] = _sha256(archive_path)
        if (
            not args.allow_source_changes
            and summary["archive_sha256"] != _EXPECTED_ARCHIVE_SHA256
        ):
            raise RuntimeError(
                "Unexpected source archive SHA-256. Expected "
                f"{_EXPECTED_ARCHIVE_SHA256}, found {summary['archive_sha256']}."
            )
        media_dir = work_dir / "source" / "media"
        videos, images = _extract_test_media(
            archive_path,
            members,
            annotations["test"][0],
            media_dir,
            workers=args.extract_workers,
        )
        _validate_test_media(media_dir, videos, images, workers=args.verify_workers)
        summary["validated_test_media"] = {
            "videos": len(videos),
            "images": len(images),
        }
        summary["published_retrieval"] = _published_retrieval_summary(
            annotations["test"][0], videos, images
        )
        corpus, queries, qrels = _build_datasets(
            annotations["test"][0], media_dir, videos, images
        )

    print(json.dumps(summary, indent=2, sort_keys=True))

    summary_path = work_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote audit summary to {summary_path}")

    if args.save_to_disk:
        assert corpus is not None and queries is not None and qrels is not None
        _save_datasets(work_dir, corpus, queries, qrels)
    if args.push:
        assert corpus is not None and queries is not None and qrels is not None
        revision = _publish(args.repo_id, corpus, queries, qrels, summary, work_dir)
        print(f"Pushed {args.repo_id} at revision {revision}")


if __name__ == "__main__":
    main()
