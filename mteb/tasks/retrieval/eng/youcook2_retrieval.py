from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/YouCook2_val"
_DATASET_REVISION = "be654b591daef1382364d563f6d3d66f61f79b35"
_BIBTEX = r"""
@inproceedings{zhou2018towards,
  author = {Zhou, Luowei and Xu, Chenliang and Corso, Jason J.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  title = {Towards Automatic Learning of Procedures from Web Instructional Videos},
  year = {2018},
}
"""


def _load_youcook2(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all YouCook2 retrieval directions.

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
    if "sentence" in query_columns:
        query = query.rename_column("sentence", "text")
    if "sentence" in corpus_columns:
        corpus = corpus.rename_column("sentence", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class YouCook2V2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2V2TRetrieval",
        description=(
            "Retrieve the English sentence that describes a given cooking video "
            "clip (video only) from the YouCook2 dataset of instructional cooking "
            "videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(self, query_columns=["video"], corpus_columns=["sentence"])


class YouCook2T2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2T2VRetrieval",
        description=(
            "Retrieve the cooking video clip (video only) that matches a given "
            "English sentence from the YouCook2 dataset of instructional cooking "
            "videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(self, query_columns=["sentence"], corpus_columns=["video"])


class YouCook2VA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2VA2TRetrieval",
        description=(
            "Retrieve the English sentence that describes a given cooking video "
            "clip from the YouCook2 dataset of instructional cooking videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(
            self, query_columns=["video", "audio"], corpus_columns=["sentence"]
        )


class YouCook2T2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2T2VARetrieval",
        description=(
            "Retrieve the cooking video clip that matches a given English sentence "
            "from the YouCook2 dataset of instructional cooking videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(
            self, query_columns=["sentence"], corpus_columns=["video", "audio"]
        )


class YouCook2V2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2V2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip from the "
            "YouCook2 dataset of instructional cooking videos. Tests cross-modal "
            "alignment between video frames and audio narration."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(self, query_columns=["video"], corpus_columns=["audio"])


class YouCook2A2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2A2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track from the "
            "YouCook2 dataset of instructional cooking videos. Tests cross-modal "
            "alignment between audio narration and video frames."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(self, query_columns=["audio"], corpus_columns=["video"])


class YouCook2VT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2VT2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given cooking video clip and "
            "its sentence description from the YouCook2 dataset of instructional "
            "cooking videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text", "audio"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(
            self, query_columns=["video", "sentence"], corpus_columns=["audio"]
        )


class YouCook2AT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2AT2VRetrieval",
        description=(
            "Retrieve the cooking video clip that matches a given audio track and "
            "its sentence description from the YouCook2 dataset of instructional "
            "cooking videos."
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="at2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "text", "video"],
        date=("2018-01-01", "2018-12-31"),
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
        _load_youcook2(
            self, query_columns=["audio", "sentence"], corpus_columns=["video"]
        )
