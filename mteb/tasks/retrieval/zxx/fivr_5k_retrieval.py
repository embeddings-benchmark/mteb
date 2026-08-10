from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset, Video, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

_DATASET_PATH = "Cerru02/FIVR-5K-MTEB"
_DATASET_REVISION = "27278cba0b40e0dc764c652efb2e9eed9b997c83"
_CORPUS_SIZE = 3_188
_QUERY_COUNT = 29
_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v")
_REGIME = Literal["dsvr", "csvr", "isvr"]

_BIBTEX = r"""
@article{kordopatis2019fivr,
  author = {Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  journal = {IEEE Transactions on Multimedia},
  title = {FIVR: Fine-grained Incident Video Retrieval},
  year = {2019},
}

@inproceedings{kordopatis2019visil,
  author = {Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  title = {ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning},
  year = {2019},
}

@article{li2024videoeval,
  author = {Li, Xinhao and Huang, Zhenpeng and Wang, Jing and Li, Kunchang and Wang, Limin},
  journal = {arXiv preprint arXiv:2407.06491},
  title = {VideoEval: Comprehensive Benchmark Suite for Low-cost Evaluation of Video Foundation Model},
  year = {2024},
}
"""

_REGIME_METADATA = {
    "dsvr": {
        "name": "FIVR5KDSVRRetrieval",
        "definition": "near-duplicate and duplicate-scene videos (ND + DS)",
        "prompt": (
            "Retrieve videos that are near-duplicates or show a duplicate scene."
        ),
    },
    "csvr": {
        "name": "FIVR5KCSVRRetrieval",
        "definition": (
            "near-duplicate, duplicate-scene, and complementary-scene videos "
            "(ND + DS + CS)"
        ),
        "prompt": (
            "Retrieve videos that are near-duplicates, show a duplicate scene, "
            "or show a complementary scene from the same incident."
        ),
    },
    "isvr": {
        "name": "FIVR5KISVRRetrieval",
        "definition": ("all videos from the same incident (ND + DS + CS + IS)"),
        "prompt": (
            "Retrieve videos from the same incident, including near-duplicates, "
            "duplicate scenes, and complementary scenes."
        ),
    },
}


def _metadata(regime: _REGIME) -> TaskMetadata:
    regime_metadata = _REGIME_METADATA[regime]
    return TaskMetadata(
        name=regime_metadata["name"],
        description=(
            "Fine-grained incident video retrieval using the established FIVR-5K "
            "protocol and VideoEval's frozen manifest. The task retrieves "
            f"{regime_metadata['definition']}. The 2026 availability freeze contains "
            "29 queries and 3,188 corpus videos; unavailable source media is recorded "
            "rather than silently discarded."
        ),
        reference="https://arxiv.org/abs/1809.04094",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score=f"map_at_{_CORPUS_SIZE}",
        date=("2013-01-01", "2017-12-31"),
        domains=["News", "Web"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": regime_metadata["prompt"]},
        is_beta=True,
    )


def _find_video(video_dir: Path, video_id: str) -> Path | None:
    for suffix in _VIDEO_SUFFIXES:
        path = video_dir / f"{video_id}{suffix}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _download_with_pytubefix(video_id: str, video_dir: Path) -> Path:
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
        and int(stream.resolution.removesuffix("p")) <= 480
    ]
    stream = max(
        eligible or streams,
        key=lambda item: int((item.resolution or "0p").removesuffix("p")),
        default=None,
    )
    if stream is None:
        raise RuntimeError("no downloadable video stream")
    suffix = f".{stream.subtype or 'mp4'}"
    partial = Path(
        stream.download(output_path=video_dir, filename=f"{video_id}.partial{suffix}")
    )
    target = video_dir / f"{video_id}{suffix}"
    partial.replace(target)
    return target


def _download_with_ytdlp(video_id: str, video_dir: Path) -> Path:
    import yt_dlp  # type: ignore[import-untyped]

    options: dict[str, Any] = {
        "continuedl": True,
        "format": "bestvideo[height<=480]/best[height<=480]/bestvideo/best",
        "noplaylist": True,
        "no_warnings": True,
        "outtmpl": str(video_dir / f"{video_id}.partial.%(ext)s"),
        "quiet": True,
        "retries": 3,
        "source_address": "0.0.0.0",
    }
    node = shutil.which("node")
    if node is not None:
        options["js_runtimes"] = {"node": {"path": node}}
        options["remote_components"] = {"ejs:github"}
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    candidates = sorted(video_dir.glob(f"{video_id}.partial.*"))
    if len(candidates) != 1:
        raise RuntimeError("download did not produce exactly one media file")
    partial = candidates[0]
    target = video_dir / f"{video_id}{partial.suffix}"
    partial.replace(target)
    return target


def _download_video(video_id: str, video_dir: Path) -> tuple[str, Path]:
    existing = _find_video(video_dir, video_id)
    if existing is not None:
        return video_id, existing

    errors: list[str] = []
    for downloader in (_download_with_pytubefix, _download_with_ytdlp):
        try:
            path = downloader(video_id, video_dir)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            return video_id, path
        except Exception as error:
            errors.append(f"{downloader.__name__}: {error}")
            for partial in video_dir.glob(f"{video_id}.partial.*"):
                partial.unlink(missing_ok=True)
    raise RuntimeError("; ".join(errors))


def _video_directory(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    configured = os.environ.get("MTEB_FIVR_VIDEO_DIR")
    if configured:
        return Path(configured).expanduser()
    cache = Path(os.environ.get("MTEB_CACHE", Path.home() / ".cache" / "mteb"))
    return cache / "datasets" / "fivr-5k" / _DATASET_REVISION


def _materialize_videos(
    rows: Sequence[dict[str, Any]],
    video_dir: Path,
    workers: int,
) -> dict[str, Path]:
    video_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_video, row["id"], video_dir): row["id"]
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            video_id = futures[future]
            try:
                returned_id, path = future.result()
                paths[returned_id] = path
            except Exception as error:
                failures[video_id] = str(error)
            if completed % 100 == 0:
                logger.info("Prepared %d/%d FIVR videos", completed, len(rows))
    if failures:
        failed_ids = ", ".join(sorted(failures)[:10])
        raise RuntimeError(
            f"Failed to materialize {len(failures)} frozen FIVR videos: {failed_ids}. "
            "The benchmark corpus is frozen and will not remove them dynamically. "
            "Install mteb[video], or set MTEB_FIVR_VIDEO_DIR to a complete local "
            "mirror, and retry."
        )
    return paths


def _as_video_dataset(
    rows: Sequence[dict[str, Any]], paths: dict[str, Path]
) -> Dataset:
    return Dataset.from_dict(
        {
            "id": [row["id"] for row in rows],
            "video": [str(paths[row["id"]]) for row in rows],
        }
    ).cast_column("video", Video())


def _load_fivr(
    task: AbsTaskRetrieval,
    regime: _REGIME,
    *,
    num_proc: int | None,
    metadata_dir: str | Path | None,
    video_dir: str | Path | None,
    download_workers: int,
) -> None:
    if task.data_loaded:
        return
    if metadata_dir is not None:
        local_dir = Path(metadata_dir).expanduser()
        corpus_metadata = Dataset.from_json(str(local_dir / "corpus.jsonl"))
        query_metadata = Dataset.from_json(str(local_dir / "queries.jsonl"))
        qrels = Dataset.from_json(str(local_dir / f"{regime}-qrels.jsonl"))
    else:
        dataset_args = {
            "path": task.metadata.dataset["path"],
            "revision": task.metadata.dataset["revision"],
            "split": "test",
            "num_proc": num_proc,
        }
        corpus_metadata = load_dataset(name="corpus", **dataset_args)
        query_metadata = load_dataset(name="queries", **dataset_args)
        qrels = load_dataset(name=f"{regime}-qrels", **dataset_args)
    if len(corpus_metadata) != _CORPUS_SIZE or len(query_metadata) != _QUERY_COUNT:
        raise ValueError("FIVR metadata counts differ from the frozen task definition")

    all_rows = list(corpus_metadata) + list(query_metadata)
    paths = _materialize_videos(
        all_rows,
        video_dir=_video_directory(video_dir),
        workers=download_workers,
    )
    relevant_docs: dict[str, dict[str, int]] = {}
    for row in qrels:
        relevant_docs.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
    if set(relevant_docs) != set(query_metadata["id"]):
        raise ValueError("FIVR qrels do not cover the frozen query set")

    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=_as_video_dataset(corpus_metadata, paths),
                queries=_as_video_dataset(query_metadata, paths),
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
        }
    }
    task.data_loaded = True


class _FIVR5KTaskMixin:
    k_values = (1, 3, 5, 10, 20, 100, 1000, _CORPUS_SIZE)
    _top_k = _CORPUS_SIZE
    _regime: _REGIME

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_fivr(
            self,
            self._regime,
            num_proc=num_proc,
            metadata_dir=(
                kwargs.get("fivr_metadata_dir")
                or os.environ.get("MTEB_FIVR_METADATA_DIR")
            ),
            video_dir=kwargs.get("fivr_video_dir"),
            download_workers=int(
                kwargs.get("fivr_download_workers")
                or os.environ.get("MTEB_FIVR_DOWNLOAD_WORKERS", "8")
            ),
        )


class FIVR5KDSVRRetrieval(_FIVR5KTaskMixin, AbsTaskRetrieval):
    metadata = _metadata("dsvr")
    _regime = "dsvr"


class FIVR5KCSVRRetrieval(_FIVR5KTaskMixin, AbsTaskRetrieval):
    metadata = _metadata("csvr")
    _regime = "csvr"


class FIVR5KISVRRetrieval(_FIVR5KTaskMixin, AbsTaskRetrieval):
    metadata = _metadata("isvr")
    _regime = "isvr"
