from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@inproceedings{li2022learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {19108--19118},
  title = {Learning to answer questions in dynamic audio-visual scenarios},
  year = {2022},
}
"""

DESC_BASE = (
    "Pair classification on the MUSIC-AVQA dataset: determining whether two "
    "clips feature the same musical instrument from 22 categories. "
    "Same-instrument / different-instrument pairs are sampled with a fixed seed."
)


class MusicAVQAVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MusicAVQAVPairClassification",
        description=DESC_BASE + " Uses video only.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "zachz/MUSIC-AVQA-PC-V",
            "revision": "ed361975cbbe7954382b334b17a4fc9857ecfa54",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
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


class MusicAVQAVAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MusicAVQAVAPairClassification",
        description=DESC_BASE + " Uses synchronized video and audio.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "zachz/MUSIC-AVQA-PC-VA",
            "revision": "a950566a7da2c21f5e83246d4a11a095bdf6be0d",
        },
        type="VideoPairClassification",
        category="va2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
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
