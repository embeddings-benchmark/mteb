from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/VGGSound_AV_RETRIEVAL"
_DATASET_REVISION = "5bfb8eb997282004d1c3b322fd081b03e2011344"
_BIBTEX = r"""
@inproceedings{chen2020vggsound,
  author = {Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle = {ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  organization = {IEEE},
  title = {Vggsound: A Large-Scale Audio-Visual Dataset},
  year = {2020},
}
"""


def _load_vggsound_av(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all VGGSound-AV retrieval directions.

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
    if "video_caption" in query_columns:
        query = query.rename_column("video_caption", "text")
    if "video_caption" in corpus_columns:
        corpus = corpus.rename_column("video_caption", "text")
    if "audio_caption" in query_columns:
        query = query.rename_column("audio_caption", "text")
    if "audio_caption" in corpus_columns:
        corpus = corpus.rename_column("audio_caption", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class VGGSoundAVV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundAVV2TRetrieval",
        description=(
            "Retrieve the visual caption that describes a given video clip (video "
            "only) from the VGGSound-AV dataset, a large-scale audio-visual dataset "
            "sourced from YouTube."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2020-01-01", "2020-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vggsound_av(
            self, query_columns=["video"], corpus_columns=["video_caption"]
        )


class VGGSoundAVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundAVT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given visual caption "
            "from the VGGSound-AV dataset, a large-scale audio-visual dataset "
            "sourced from YouTube."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2020-01-01", "2020-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vggsound_av(
            self, query_columns=["video_caption"], corpus_columns=["video"]
        )


class VGGSoundAVVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundAVVA2TRetrieval",
        description=(
            "Retrieve the visual caption that describes a given video+audio clip "
            "from the VGGSound-AV dataset, a large-scale audio-visual dataset "
            "sourced from YouTube."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2020-01-01", "2020-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vggsound_av(
            self, query_columns=["video", "audio"], corpus_columns=["video_caption"]
        )


class VGGSoundAVT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundAVT2VARetrieval",
        description=(
            "Retrieve the video+audio clip that matches a given visual caption "
            "from the VGGSound-AV dataset, a large-scale audio-visual dataset "
            "sourced from YouTube."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2020-01-01", "2020-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vggsound_av(
            self, query_columns=["video_caption"], corpus_columns=["video", "audio"]
        )
