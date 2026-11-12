from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
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


class MELDAudioVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MELDAudioVideoZeroShot",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear Sampled the test split (~2,610 examples).",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELDAudioVideoZeroShot",
            "revision": "d1ed5639cc3262d1be68558ef41eda48b51dd943",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment", "Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=CITATION,
    )


class MELDVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MELDVideoZeroShot",
        description="MELD (Multimodal EmotionLines Dataset) is a multimodal emotion recognition dataset containing over 13,000 utterances from the Friends TV series, labeled with 7 emotion categories: Anger, Disgust, Sadness, Joy, Neutral, Surprise, and Fear Sampled the test split (~2,610 examples).",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "mteb/MELDVideoZeroShot",
            "revision": "29ad15781b8758a051b6fa02e18bbe44012dce64",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment", "Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
    )
