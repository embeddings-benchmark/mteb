from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from datasets import concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

_DATASET_PATH = "Cerru02/LombardGrid-Retrieval"
_DATASET_REVISION = "a21a537577170e6912c3b6ab0d593e1312562c03"

_BIBTEX = r"""
@article{alghamdi2018corpus,
  author = {Alghamdi, Najwa and Maddock, Steve and Marxer, Ricard and Barker, Jon and Brown, Guy J.},
  doi = {10.1121/1.5042758},
  journal = {The Journal of the Acoustical Society of America},
  number = {6},
  pages = {EL523--EL529},
  title = {A corpus of audio-visual Lombard speech with frontal and profile views},
  volume = {143},
  year = {2018},
}
"""

_DESCRIPTION = (
    "Utterance retrieval derived from the Lombard GRID audio-visual speech corpus. "
    "The frozen protocol contains ten sentence codes for each of 54 speakers. All "
    "relevance judgments link recordings of the same utterance rather than recordings "
    "that merely share a speaker. The source paper introduced the corpus, not these "
    "retrieval tasks."
)

_COMMON = {
    "reference": "https://doi.org/10.1121/1.5042758",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyRetrieval",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "main_score": "ndcg_at_10",
    # The source does not report collection dates; use the publication year.
    "date": ("2018-01-01", "2018-12-31"),
    "domains": ["Spoken"],
    "license": "cc-by-4.0",
    "annotations_creators": "derived",
    "dialect": ["eng-GB"],
    "sample_creation": "created",
    "is_beta": True,
    "bibtex_citation": _BIBTEX,
}


def _ordered_select(dataset: Dataset, column: str, ids: list[str]) -> Dataset:
    row_by_id = {id_: index for index, id_ in enumerate(dataset[column])}
    missing = sorted(set(ids) - row_by_id.keys())
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} referenced IDs")
    return dataset.select([row_by_id[id_] for id_ in ids])


def _audio(dataset: Dataset) -> Dataset:
    return dataset.select_columns(["audio_id", "audio"]).rename_column("audio_id", "id")


def _video(dataset: Dataset, view: Literal["front", "profile"]) -> Dataset:
    return dataset.select_columns([f"{view}_video_id", f"{view}_video"]).rename_columns(
        {f"{view}_video_id": "id", f"{view}_video": "video"}
    )


def _video_audio(dataset: Dataset) -> Dataset:
    return dataset.select_columns(
        ["recording_id", "front_video", "audio"]
    ).rename_columns({"recording_id": "id", "front_video": "video"})


def _load_lombard_grid(
    task: AbsTaskRetrieval,
    direction: Literal["a2v", "v2a", "v2v", "va2va"],
) -> None:
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    media = load_dataset(
        _DATASET_PATH, "media", revision=_DATASET_REVISION, split=split
    )

    if direction == "va2va":
        links = load_dataset(
            _DATASET_PATH,
            "condition_pairs",
            revision=_DATASET_REVISION,
            split=split,
        ).to_dict()
        plain_ids = links["plain_recording_id"]
        lombard_ids = links["lombard_recording_id"]
        queries = _video_audio(_ordered_select(media, "recording_id", plain_ids))
        corpus = _video_audio(_ordered_select(media, "recording_id", lombard_ids))
        qrels = {
            plain_id: {lombard_id: 1}
            for plain_id, lombard_id in zip(plain_ids, lombard_ids, strict=True)
        }
    else:
        links = load_dataset(
            _DATASET_PATH,
            "matching_utterances",
            revision=_DATASET_REVISION,
            split=split,
        ).to_dict()
        audio_ids = links["audio_id"]
        front_ids = links["front_video_id"]
        profile_ids = links["profile_video_id"]
        selected_media = _ordered_select(media, "audio_id", audio_ids)
        audio = _audio(selected_media)
        front = _video(selected_media, "front")
        profile = _video(selected_media, "profile")

        if direction == "a2v":
            queries = audio
            corpus = concatenate_datasets([front, profile])
            qrels = {
                audio_id: {front_id: 1, profile_id: 1}
                for audio_id, front_id, profile_id in zip(
                    audio_ids, front_ids, profile_ids, strict=True
                )
            }
        elif direction == "v2a":
            queries = concatenate_datasets([front, profile])
            corpus = audio
            qrels = {
                video_id: {audio_id: 1}
                for audio_id, front_id, profile_id in zip(
                    audio_ids, front_ids, profile_ids, strict=True
                )
                for video_id in (front_id, profile_id)
            }
        else:
            queries = front
            corpus = profile
            qrels = {
                front_id: {profile_id: 1}
                for front_id, profile_id in zip(front_ids, profile_ids, strict=True)
            }

    task.dataset = {
        "default": {
            split: RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        }
    }
    task.data_loaded = True


class LombardGridA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LombardGridA2VRetrieval",
        description=(
            f"{_DESCRIPTION} An audio query retrieves both the frontal and profile "
            "videos of the same recording. There are 540 queries, 1,080 corpus "
            "videos, and two positives per query."
        ),
        category="a2v",
        modalities=["audio", "video"],
        task_subtypes=["Cross-Modal Retrieval", "Speech Retrieval"],
        prompt={
            "query": "Retrieve the frontal and profile videos matching this audio utterance."
        },
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_lombard_grid(self, "a2v")


class LombardGridV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LombardGridV2ARetrieval",
        description=(
            f"{_DESCRIPTION} Each frontal or profile video query retrieves the audio "
            "from the same recording. There are 1,080 queries, 540 corpus audio "
            "items, and one positive per query."
        ),
        category="v2a",
        modalities=["video", "audio"],
        task_subtypes=["Cross-Modal Retrieval", "Speech Retrieval"],
        prompt={"query": "Retrieve the audio matching this video utterance."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_lombard_grid(self, "v2a")


class LombardGridV2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LombardGridV2VRetrieval",
        description=(
            f"{_DESCRIPTION} A frontal-view query retrieves the synchronized "
            "profile-view video of the same recording. There are 540 queries, 540 "
            "corpus videos, and one positive per query."
        ),
        category="v2v",
        modalities=["video"],
        task_subtypes=["Speech Retrieval"],
        prompt={
            "query": "Retrieve the profile-view video matching this frontal-view utterance."
        },
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_lombard_grid(self, "v2v")


class LombardGridVA2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LombardGridVA2VARetrieval",
        description=(
            f"{_DESCRIPTION} A plain-speech frontal video with its separate audio "
            "retrieves the Lombard-condition frontal video and audio for the same "
            "speaker and sentence code. Fixing the camera view isolates the speech "
            "condition change. There are 540 queries, 540 corpus items, and one "
            "positive per query."
        ),
        category="va2va",
        modalities=["video", "audio"],
        task_subtypes=["Cross-Modal Retrieval", "Speech Retrieval"],
        prompt={
            "query": (
                "Retrieve the Lombard video and audio matching this plain-speech "
                "video and audio utterance."
            )
        },
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_lombard_grid(self, "va2va")
