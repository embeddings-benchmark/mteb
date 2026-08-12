from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset, Video, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def _video_directory(override: str | Path | None = None) -> Path:
    configured = override or os.environ.get("MTEB_FIVR_VIDEO_DIR")
    if configured is None:
        raise ValueError(
            "FIVR media is not distributed with the metadata dataset. Prepare a "
            "local video directory with scripts/data/fivr/create_data.py, then set "
            "MTEB_FIVR_VIDEO_DIR or pass fivr_video_dir to load_data()."
        )
    video_dir = Path(configured).expanduser()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"FIVR video directory does not exist: {video_dir}")
    return video_dir


def _resolve_local_videos(
    rows: Sequence[dict[str, Any]], video_dir: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for row in rows:
        video_id = row["id"]
        path = _find_video(video_dir, video_id)
        if path is None:
            missing.append(video_id)
        else:
            paths[video_id] = path
    if missing:
        missing_ids = ", ".join(sorted(missing)[:10])
        raise FileNotFoundError(
            f"FIVR video directory is missing {len(missing)} frozen videos: "
            f"{missing_ids}. The task never downloads media or changes its frozen "
            "corpus; prepare the complete directory with "
            "scripts/data/fivr/create_data.py and retry."
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
    paths = _resolve_local_videos(all_rows, _video_directory(video_dir))
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
