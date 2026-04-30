from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/DiDeMo"
_DATASET_REVISION = "746689f644b66022540a9a39136e842bee164e6b"
_BIBTEX = r"""
@inproceedings{hendricks2017localizing,
  author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  title = {Localizing Moments in Video with Natural Language},
  year = {2017},
}
"""


def _load_didemo(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all DiDeMo retrieval directions.

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


class DiDeMoV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoV2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip (video "
            "only) from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["video", "text"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["video"], corpus_columns=["caption"])


class DiDeMoT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given English "
            "caption from the DiDeMo dataset of Flickr videos with temporally "
            "grounded sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["text", "video"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["caption"], corpus_columns=["video"])


class DiDeMoVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoVA2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip "
            "from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["audio", "video", "text"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["video", "audio"], corpus_columns=["caption"])


class DiDeMoT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoT2VARetrieval",
        description=(
            "Retrieve the video clip that matches a given English caption "
            "from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["text", "audio", "video"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["caption"], corpus_columns=["video", "audio"])


class DiDeMoV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoV2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip from the "
            "DiDeMo dataset. Tests cross-modal alignment between video frames and audio."
        ),
        reference="https://arxiv.org/abs/1708.01355",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2017-01-01", "2017-12-31"),
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
        _load_didemo(self, query_columns=["video"], corpus_columns=["audio"])


class DiDeMoA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoA2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track from the "
            "DiDeMo dataset. Tests cross-modal alignment between audio and video frames."
        ),
        reference="https://arxiv.org/abs/1708.01355",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2017-01-01", "2017-12-31"),
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
        _load_didemo(self, query_columns=["audio"], corpus_columns=["video"])


class DiDeMoVT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoVT2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip and its caption "
            "from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["video", "text", "audio"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the audio that corresponds to the following video and its description."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["video", "caption"], corpus_columns=["audio"])


class DiDeMoAT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoAT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track and its caption "
            "from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="at2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["audio", "text", "video"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the video that corresponds to the following audio and its description."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_didemo(self, query_columns=["audio", "caption"], corpus_columns=["video"])
