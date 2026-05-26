from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@inproceedings{poria2019meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages = {527--536},
  title = {Meld: A multimodal multi-party dataset for emotion recognition in conversations},
  year = {2019},
}
"""

DESC_BASE = (
    "Pair classification on the MELD dataset: determining whether two utterance "
    "clips from the Friends TV series express the same emotion from 7 categories "
    "(anger, disgust, fear, joy, neutral, sadness, surprise). Same-emotion / "
    "different-emotion pairs are sampled with a fixed seed."
)


class MELDVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MELDVPairClassification",
        description=DESC_BASE + " Uses video only.",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "zachz/MELD-PC-V",
            "revision": "3ae0531412ac647748f9f644cf07694f5c140fcd",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment", "Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"


class MELDVAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MELDVAPairClassification",
        description=DESC_BASE + " Uses synchronized video and audio.",
        reference="https://aclanthology.org/P19-1050.pdf",
        dataset={
            "path": "zachz/MELD-PC-VA",
            "revision": "487f8f9d0f43331c8bee469746e84e665e733dff",
        },
        type="VideoPairClassification",
        category="va2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2019-01-01", "2019-07-28"),
        domains=["Entertainment", "Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name = (("video1", "video"), ("audio1", "audio"))
    input2_column_name = (("video2", "video"), ("audio2", "audio"))
    label_column_name: str = "label"
