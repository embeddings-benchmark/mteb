from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MELDClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MELDClassification",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELD",
            "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),  # around time of conference
        domains=["Entertainment"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{poria2019meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages = {527--536},
  title = {Meld: A multimodal multi-party dataset for emotion recognition in conversations},
  year = {2019},
}
""",
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "emotion"
    is_cross_validation: bool = True
    train_split: str = "test"

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )


class MELDVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MELDVideoClassification",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELD",
            "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),  # around time of conference
        domains=["Entertainment"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{poria2019meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages = {527--536},
  title = {Meld: A multimodal multi-party dataset for emotion recognition in conversations},
  year = {2019},
}
""",
    )
    input_column_name = "video"
    label_column_name: str = "emotion"
    is_cross_validation: bool = True
    train_split: str = "test"

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
