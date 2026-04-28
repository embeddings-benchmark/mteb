from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@inproceedings{poria2019meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages = {527--536},
  title = {Meld: A multimodal multi-party dataset for emotion recognition in conversations},
  year = {2019},
}
"""


class MELDAudioVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MELDAudioVideoZeroShot",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELD",
            "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=CITATION,
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "emotion"

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name, n_samples=2048
        )

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]


class MELDVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MELDVideoZeroShot",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELD",
            "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=CITATION,
    )
    input_column_name = "video"
    label_column_name: str = "emotion"

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name, n_samples=2048
        )

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]