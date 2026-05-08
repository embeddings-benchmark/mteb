from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/AudioCaps_AV"
_DATASET_REVISION = "5fd062d13a1ea8cdf36923faaacc32c43e6de14e"
_BIBTEX = r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {Audiocaps: Generating captions for audios in the wild},
  year = {2019},
}
"""


def _load_audiocaps_av(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all AudioCaps-AV retrieval directions.

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


class AudioCapsAVV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVV2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip (video "
            "only) from AudioCaps-AV, an audio-visual extension of the AudioCaps "
            "dataset sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(self, query_columns=["video"], corpus_columns=["caption"])


class AudioCapsAVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given English "
            "caption from AudioCaps-AV, an audio-visual extension of the AudioCaps "
            "dataset sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(self, query_columns=["caption"], corpus_columns=["video"])


class AudioCapsAVVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVVA2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video+audio clip "
            "from AudioCaps-AV, an audio-visual extension of the AudioCaps dataset "
            "sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(
            self, query_columns=["video", "audio"], corpus_columns=["caption"]
        )


class AudioCapsAVT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVT2VARetrieval",
        description=(
            "Retrieve the video+audio clip that matches a given English caption "
            "from AudioCaps-AV, an audio-visual extension of the AudioCaps dataset "
            "sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(
            self, query_columns=["caption"], corpus_columns=["video", "audio"]
        )


class AudioCapsAVV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVV2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip from "
            "AudioCaps-AV, an audio-visual extension of the AudioCaps dataset "
            "sourced from YouTube. Tests cross-modal alignment between video "
            "frames and audio."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(self, query_columns=["video"], corpus_columns=["audio"])


class AudioCapsAVA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVA2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track from "
            "AudioCaps-AV, an audio-visual extension of the AudioCaps dataset "
            "sourced from YouTube. Tests cross-modal alignment between audio "
            "and video frames."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_audiocaps_av(self, query_columns=["audio"], corpus_columns=["video"])


class AudioCapsAVVT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVVT2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip and its "
            "caption from AudioCaps-AV, an audio-visual extension of the AudioCaps "
            "dataset sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text", "audio"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
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
        _load_audiocaps_av(
            self, query_columns=["video", "caption"], corpus_columns=["audio"]
        )


class AudioCapsAVAT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsAVAT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track and its "
            "caption from AudioCaps-AV, an audio-visual extension of the AudioCaps "
            "dataset sourced from YouTube."
        ),
        reference="https://audiocaps.github.io/",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="at2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "text", "video"],
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
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
        _load_audiocaps_av(
            self, query_columns=["audio", "caption"], corpus_columns=["video"]
        )
