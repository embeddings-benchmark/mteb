from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{poria2019meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages = {527--536},
  title = {Meld: A multimodal multi-party dataset for emotion recognition in conversations},
  year = {2019},
}
"""

DESCRIPTION = (
    "MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition "
    "dataset containing over 13,000 utterances from the Friends TV series, "
    "labeled with 7 emotion categories and 100 speaker identities."
)

DATASET = {
    "path": "mteb/MELD",
    "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
}

REFERENCE = "https://aclanthology.org/P19-1050.pdf"
DATE = ("2019-01-01", "2019-07-28")


EMOTION_DESCRIPTION_SUFFIX = (
    " This task clusters by 6 emotion categories (Anger, Disgust, Fear, Joy, "
    "Sadness, Surprise); the dominant Neutral class is excluded to mitigate "
    "class imbalance."
)


class MELDEmotionAudioVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MELDEmotionAudioVideoClustering",
        description=DESCRIPTION + EMOTION_DESCRIPTION_SUFFIX,
        reference=REFERENCE,
        dataset=DATASET,
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=DATE,
        domains=["Entertainment"],
        task_subtypes=["Emotion Clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "emotion"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            neutral_id = ds.features["emotion"].str2int("neutral")
            ds = ds.filter(lambda x: x["emotion"] != neutral_id)
            self.dataset[split] = ds.select_columns(["video", "audio", "emotion"])


class MELDEmotionVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MELDEmotionVideoClustering",
        description=DESCRIPTION + EMOTION_DESCRIPTION_SUFFIX,
        reference=REFERENCE,
        dataset=DATASET,
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=DATE,
        domains=["Entertainment"],
        task_subtypes=["Emotion Clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "emotion"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            neutral_id = ds.features["emotion"].str2int("neutral")
            ds = ds.filter(lambda x: x["emotion"] != neutral_id)
            self.dataset[split] = ds.select_columns(["video", "emotion"])


class MELDSpeakerAudioVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MELDSpeakerAudioVideoClustering",
        description=DESCRIPTION + " This task clusters by speaker identity.",
        reference=REFERENCE,
        dataset=DATASET,
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=DATE,
        domains=["Entertainment"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "speaker"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "audio", "speaker"],
            )


class MELDSpeakerVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MELDSpeakerVideoClustering",
        description=DESCRIPTION + " This task clusters by speaker identity.",
        reference=REFERENCE,
        dataset=DATASET,
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=DATE,
        domains=["Entertainment"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "speaker"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "speaker"],
            )
