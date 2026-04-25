from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/VATEX_test_1k"
_DATASET_REVISION = "0d2e86e6d36927f4676ee6127c4e38e3867ce0ce"
_BIBTEX = r"""
@inproceedings{wang2019vatex,
  author = {Wang, Xin and Wu, Jiawei and Chen, Junkun and Li, Lei and Wang, Yuan-Fang and Wang, William Yang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  title = {VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research},
  year = {2019},
}
"""


def _load_vatex(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all VATEX retrieval directions.

    TODO: Reupload dataset in standard format and remove this custom load_data.
    """
    if task.data_loaded:
        return
    task.dataset = {"default": {}}
    dataset = load_dataset(
        task.metadata.dataset["path"],
        revision=task.metadata.dataset["revision"],
        split=task.metadata.eval_splits[0],
    )
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

    query = dataset.select_columns(["id"] + query_columns)
    corpus = dataset.select_columns(["id"] + corpus_columns)
    if "caption" in query_columns:
        query = query.rename_column("caption", "text")
    if "caption" in corpus_columns:
        corpus = corpus.rename_column("caption", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class VATEXV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXV2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip (video "
            "only) from VATEX, a large-scale multilingual video description dataset."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["video"], corpus_columns=["caption"])


class VATEXT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given English "
            "caption from VATEX, a large-scale multilingual video description dataset."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["caption"], corpus_columns=["video"])


class VATEXVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXVA2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip from "
            "VATEX, a large-scale multilingual video description dataset."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["video", "audio"], corpus_columns=["caption"])


class VATEXT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXT2VARetrieval",
        description=(
            "Retrieve the video clip that matches a given English caption from "
            "VATEX, a large-scale multilingual video description dataset."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["caption"], corpus_columns=["video", "audio"])


class VATEXV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXV2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip from the "
            "VATEX dataset. Tests cross-modal alignment between video frames and audio."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["video"], corpus_columns=["audio"])


class VATEXA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXA2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track from the "
            "VATEX dataset. Tests cross-modal alignment between audio and video frames."
        ),
        reference="https://arxiv.org/abs/1904.03493",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vatex(self, query_columns=["audio"], corpus_columns=["video"])
