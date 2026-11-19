from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/MomentSeekerChunked"
_DATASET_REVISION = "f1195925fd67ef2d676e42ffa098c4a8ec8508e9"
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
    "MomentSeeker long-video moment retrieval under the paper's protocol. Each long "
    "video (1,200 s on average; movies, cartoons, anomaly, egocentric and sports "
    "footage) is cut into fixed 10-second chunks, and a query ranks the chunks of its "
    "own video. A chunk is relevant if its temporal IoU with an annotated answer "
    "moment is at least 0.3. Queries for which no chunk reaches that threshold are "
    "kept and score 0, as in the paper. "
)


def _load_chunked(task: AbsTaskRetrieval, subset: str) -> None:
    """Load one direction-level subset, restricting the shared chunk corpus to its candidates."""
    if task.data_loaded:
        return
    path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    top_ranked_ds = load_dataset(
        path, f"{subset}-top_ranked", split="test", revision=revision
    )
    top_ranked = {row["query-id"]: row["corpus-ids"] for row in top_ranked_ds}
    candidates = {cid for ids in top_ranked.values() for cid in ids}
    corpus = load_dataset(path, "corpus", split="test", revision=revision)
    corpus = corpus.rename_column("_id", "id")
    keep = [i for i, cid in enumerate(corpus["id"]) if cid in candidates]
    corpus = corpus.select(keep)
    queries = load_dataset(path, f"{subset}-queries", split="test", revision=revision)
    qrels_ds = load_dataset(path, f"{subset}-qrels", split="test", revision=revision)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        # score-0 rows keep queries with no IoU>=0.3 chunk in the evaluation
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    task.dataset = {
        "default": {
            "test": RetrievalSplitData(
                corpus=corpus,
                queries=queries,
                relevant_docs=qrels,
                top_ranked=top_ranked,
            )
        }
    }
    task.data_loaded = True


class MomentSeekerT2VGlobalLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VGlobalLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries are text questions. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
        prompt={
            "query": "Given the question, retrieve the video segment that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "t2v-global")


class MomentSeekerTI2VGlobalLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VGlobalLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "ti2v-global")


class MomentSeekerTV2VGlobalLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VGlobalLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers global-level moments (Causal Reasoning, Spatial Relation)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "tv2v-global")


class MomentSeekerT2VEventLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VEventLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries are text questions. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
        prompt={
            "query": "Given the question, retrieve the video segment that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "t2v-event")


class MomentSeekerTI2VEventLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VEventLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "ti2v-event")


class MomentSeekerTV2VEventLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VEventLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers event-level moments (Description Location, Action Recognition, Anomaly Detection)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "tv2v-event")


class MomentSeekerT2VObjectLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerT2VObjectLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries are text questions. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
        prompt={
            "query": "Given the question, retrieve the video segment that answers it."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "t2v-object")


class MomentSeekerTI2VObjectLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTI2VObjectLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference image. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "ti2v-object")


class MomentSeekerTV2VObjectLevelReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerTV2VObjectLevelReranking",
        description=(
            _DESCRIPTION
            + "Queries combine a text question with a reference video clip. This subtask covers object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR)."
        ),
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyReranking",
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
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_chunked(self, "tv2v-object")
