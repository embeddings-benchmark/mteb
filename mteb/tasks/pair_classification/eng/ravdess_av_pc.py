from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
"""

DESC_BASE = (
    "Pair classification on the RAVDESS dataset: determining whether two acted "
    "speech/song clips express the same emotion from 8 categories (neutral, calm, "
    "happy, sad, angry, fearful, surprise, disgust). Same-emotion / "
    "different-emotion pairs are sampled with a fixed seed."
)


class RAVDESSAVVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RAVDESSAVVPairClassification",
        description=DESC_BASE + " Uses video only.",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "zachz/RAVDESS-AV-PC-V",
            "revision": "feb9e4c2f1b147c8eb3b9861a65b3f34a7073e88",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"


class RAVDESSAVVAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RAVDESSAVVAPairClassification",
        description=DESC_BASE + " Uses synchronized video and audio.",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "zachz/RAVDESS-AV-PC-VA",
            "revision": "9e33e9bb22fd1df68ee4ace9210fd375f9e8f8f7",
        },
        type="VideoPairClassification",
        category="va2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name = (("video1", "video"), ("audio1", "audio"))
    input2_column_name = (("video2", "video"), ("audio2", "audio"))
    label_column_name: str = "label"
