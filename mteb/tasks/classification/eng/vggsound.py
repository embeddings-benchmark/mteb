from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{chen2020vggsound,
  author = {Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle = {ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/ICASSP40776.2020.9053174},
  organization = {IEEE},
  pages = {721-725},
  title = {VGGSound: A Large-Scale Audio-Visual Dataset},
  year = {2020},
}
"""


class VGGSoundVAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VGGSoundVA",
        description="VGGSound is a large-scale audio-visual dataset of short YouTube clips spanning 308 sound classes (e.g. 'playing piano', 'dog barking'). Audio is the primary signal. This variant uses both video and audio modalities.",
        reference="https://arxiv.org/abs/2004.14368",
        dataset={
            "path": "mteb/VGGSound",
            "revision": "a994211ca0996558ff6cbd6977b4c1749f49c889",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-05-04",
            "2020-05-08",
        ),
        domains=["Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True


class VGGSoundVClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VGGSoundV",
        description="VGGSound is a large-scale audio-visual dataset of short YouTube clips spanning 308 sound classes (e.g. 'playing piano', 'dog barking'). This variant uses video only as a baseline; audio is the primary signal in the original task.",
        reference="https://arxiv.org/abs/2004.14368",
        dataset={
            "path": "mteb/VGGSound",
            "revision": "a994211ca0996558ff6cbd6977b4c1749f49c889",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-05-04",
            "2020-05-08",
        ),
        domains=["Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = "video"
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True
