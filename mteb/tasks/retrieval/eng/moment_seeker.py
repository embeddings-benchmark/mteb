from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "dukesun99/MomentSeeker-CVR"
_DATASET_REVISION = "e87fc71839cd14dfd09aa83b378cd51074318b3c"
_REFERENCE = "https://arxiv.org/abs/2502.12558"
_BIBTEX = r"""
@inproceedings{yuan2025momentseeker,
  author = {Yuan, Huaying and Ni, Jian and Liu, Zheng and Wang, Yueze and Zhou, Junjie and Liang, Zhengyang and Zhao, Bo and Cao, Zhao and Wen, Ji-Rong and Dou, Zhicheng},
  booktitle = {Advances in Neural Information Processing Systems 38},
  collection = {NeurIPS 2025},
  doi = {10.52202/085713-0934},
  pages = {31491–31517},
  publisher = {Neural Information Processing Systems Foundation, Inc. (NeurIPS)},
  series = {NeurIPS 2025},
  title = {MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval},
  url = {https://doi.org/10.52202/085713-0934},
  year = {2025},
}
"""
_DESCRIPTION = (
    "Composed video retrieval adapted from MomentSeeker, a long-video moment "
    "retrieval benchmark with videos averaging over 500 seconds (movies, "
    "cartoons, egocentric and open-domain footage). Source videos are split into "
    "30-second non-overlapping 360p chunks forming the corpus; a chunk is "
    "relevant to a query iff it overlaps an annotated answer interval by at "
    "least half of the shorter of chunk and interval. "
)


def _load_momentseeker(task: AbsTaskRetrieval, direction: str) -> None:
    """Load the shared chunk corpus plus the direction's queries and qrels."""
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    queries = load_dataset(
        path, f"{direction}-queries", split="test", revision=revision
    )
    qrels_ds = load_dataset(path, f"{direction}-qrels", split="test", revision=revision)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None
            )
        }
    }
    task.data_loaded = True


_FULL_DATASET_PATH = "dukesun99/MomentSeeker-Full"
_FULL_DATASET_REVISION = "e1be36f6a27a0da618704860cb50517634c01f3a"
_FULL_DESCRIPTION = (
    "MomentSeeker full-video retrieval. The corpus is the 271 complete source videos "
    "(1,200 s on average, re-encoded to 360p with audio), and a video is relevant to "
    "a query if it contains the annotated answer moment. This is a video-identification "
    "variant of the benchmark; the MomentSeeker*Reranking tasks follow the paper's "
    "within-video chunk protocol. "
)


def _load_full(task: AbsTaskRetrieval, subset: str) -> None:
    """Load the shared full-video corpus plus one direction-level subset."""
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    queries = load_dataset(path, f"{subset}-queries", split="test", revision=revision)
    qrels_ds = load_dataset(path, f"{subset}-qrels", split="test", revision=revision)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None
            )
        }
    }
    task.data_loaded = True


class MomentSeekerTI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VRetrieval",
        description=_DESCRIPTION
        + "Queries combine a text question with a reference image, e.g. asking "
        "what happened around the pictured moment.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference image, retrieve the video segment that answers it."
        },
        is_beta=True,
        superseded_by="MomentSeekerReranking",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_momentseeker(self, "ti2v")


class MomentSeekerTV2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VRetrieval",
        description=_DESCRIPTION
        + "Queries combine a text question with a reference video clip, e.g. "
        "asking what happened after the shown event.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference clip, retrieve the video segment that answers it."
        },
        is_beta=True,
        superseded_by="MomentSeekerReranking",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_momentseeker(self, "tv2v")


class MomentSeekerT2VGlobalLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VGlobalLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries are text questions. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Given the question, retrieve the video that answers it."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "t2v-global")


class MomentSeekerTI2VGlobalLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VGlobalLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference image, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "ti2v-global")


class MomentSeekerTV2VGlobalLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VGlobalLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference clip, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "tv2v-global")


class MomentSeekerT2VEventLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VEventLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries are text questions. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Given the question, retrieve the video that answers it."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "t2v-event")


class MomentSeekerTI2VEventLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VEventLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference image, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "ti2v-event")


class MomentSeekerTV2VEventLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VEventLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference clip, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "tv2v-event")


class MomentSeekerT2VObjectLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VObjectLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries are text questions. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Given the question, retrieve the video that answers it."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "t2v-object")


class MomentSeekerTI2VObjectLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VObjectLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference image, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "ti2v-object")


class MomentSeekerTV2VObjectLevelFullVideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VObjectLevelFullVideoRetrieval",
        description=(
            _FULL_DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _FULL_DATASET_PATH, "revision": _FULL_DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_5",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Given the question and the reference clip, retrieve the video that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_full(self, "tv2v-object")
