from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@inproceedings{tian2018audio,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {Proceedings of the European conference on computer vision (ECCV)},
  pages = {247--263},
  title = {Audio-visual event localization in unconstrained videos},
  year = {2018},
}
"""

DESC_BASE = (
    "Pair classification on the Audio-Visual Event (AVE) dataset: determining "
    "whether two short clips contain the same audio-visual event from 28 "
    "categories (e.g. accordion, guitar, helicopter, speech). Same-event / "
    "different-event pairs are sampled with a fixed seed."
)


class AVEDatasetVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AVEDatasetVPairClassification",
        description=DESC_BASE + " Uses video only.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "zachz/AVE-Dataset-PC-V",
            "revision": "53fc1bb0fb8c7dd09425e25fc2936712c1098e99",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-01-01", "2018-09-01"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Environment Sound Classification"],
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


class AVEDatasetVAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AVEDatasetVAPairClassification",
        description=DESC_BASE + " Uses synchronized video and audio.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "zachz/AVE-Dataset-PC-VA",
            "revision": "322c081c78af9f771e3ef52b27c30b47056db41a",
        },
        type="VideoPairClassification",
        category="va2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-01-01", "2018-09-01"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Environment Sound Classification"],
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
