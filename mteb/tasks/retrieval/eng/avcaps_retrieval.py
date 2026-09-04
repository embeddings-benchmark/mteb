from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/avcaps-retrieval"
_DATASET_REVISION = "adda52e701ac328a63ab4cfbdfcf1ccdd593939f"

_BIBTEX = r"""
@article{sudarsanam2025avcaps,
  author = {Sudarsanam, Parthasaarathy and Martin-Morato, Irene and Hakala, Aapo and Virtanen, Tuomas},
  journal = {IEEE Open Journal of Signal Processing},
  pages = {691--704},
  title = {{AVCaps}: An Audio-Visual Dataset with Modality-Specific Captions},
  volume = {6},
  year = {2025},
}
"""

_DESCRIPTION = (
    "Audio-visual retrieval from AVCaps, derived from VidOR. Each clip is captioned three "
    "separate ways - from the audio alone, from the visuals alone, and from both together "
    "- so the audio-only, video-only, and combined directions can be scored independently "
    "on identical clips rather than inferred from a single caption set."
)

_COMMON = {
    "reference": "https://ieeexplore.ieee.org/document/11029114/",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyRetrieval",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "main_score": "ndcg_at_10",
    "date": ("2024-01-01", "2025-06-30"),
    "domains": ["Web", "AudioScene"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-nc-4.0",
    "annotations_creators": "human-annotated",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}

# caption config -> the media columns that caption is allowed to describe
_CAPTION_MEDIA = {
    "audio_captions": ["audio"],
    "visual_captions": ["video"],
    "av_captions": ["video", "audio"],
}


def _load_avcaps(task: AbsTaskRetrieval, config: str, to_text: bool) -> None:
    """Load one AVCaps direction.

    `to_text` selects media->caption; otherwise caption->media. Only the media columns
    the caption actually describes are exposed, so an audio-caption task cannot be
    solved from the video track.
    """
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    cols = _CAPTION_MEDIA[config]
    media = load_dataset(
        _DATASET_PATH, "media", revision=_DATASET_REVISION, split=split
    ).select_columns(["id", *cols])
    caps = load_dataset(_DATASET_PATH, config, revision=_DATASET_REVISION, split=split)
    text_ds = caps.select_columns(["id", "text"])

    # Read the link columns directly; iterating full rows would decode the media.
    links = caps.select_columns(["id", "media_id"]).to_dict()
    pairs = list(zip(links["id"], links["media_id"], strict=True))

    if to_text:
        qrels: dict[str, dict[str, int]] = {}
        for cid, mid in pairs:
            qrels.setdefault(mid, {})[cid] = 1
        keep = set(qrels)
        # select() by index rather than filter(), which would decode every clip
        wanted = [i for i, id_ in enumerate(media["id"]) if id_ in keep]
        queries = media.select(wanted)
        corpus = text_ds
    else:
        queries = text_ds
        corpus = media
        qrels = {cid: {mid: 1} for cid, mid in pairs}

    task.dataset = {
        "default": {
            split: RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        }
    }
    task.data_loaded = True


class AVCapsA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the caption describing a clip's audio track.",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the caption that describes this audio."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "audio_captions", to_text=True)


class AVCapsT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the audio track matching a caption of it.",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the audio that matches the following description."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "audio_captions", to_text=False)


class AVCapsV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsV2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the caption describing a clip's visuals.",
        category="v2t",
        modalities=["video", "text"],
        prompt={"query": "Find the caption that describes this video."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "visual_captions", to_text=True)


class AVCapsT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsT2VRetrieval",
        description=f"{_DESCRIPTION} Retrieve the video matching a caption of its visuals.",
        category="t2v",
        modalities=["text", "video"],
        prompt={"query": "Find the video that matches the following description."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "visual_captions", to_text=False)


class AVCapsVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsVA2TRetrieval",
        description=(
            f"{_DESCRIPTION} Retrieve the caption describing a clip's video and audio together."
        ),
        category="va2t",
        modalities=["video", "audio", "text"],
        prompt={"query": "Find the caption that describes this video and its audio."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "av_captions", to_text=True)


class AVCapsT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVCapsT2VARetrieval",
        description=(
            f"{_DESCRIPTION} Retrieve the video and audio matching a combined caption."
        ),
        category="t2va",
        modalities=["text", "video", "audio"],
        prompt={
            "query": "Find the video and audio matching the following description."
        },
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_avcaps(self, "av_captions", to_text=False)
