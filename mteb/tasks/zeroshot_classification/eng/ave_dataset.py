from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)

CITATION = r"""
@inproceedings{tian2018audio,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {Proceedings of the European conference on computer vision (ECCV)},
  pages = {247--263},
  title = {Audio-visual event localization in unconstrained videos},
  year = {2018},
}
"""


class AVEDatasetZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="AVEDatasetZeroShot",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from synchronized video and audio. This variant uses both video and audio modalities.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVE-Dataset",
            "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-09-01"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a video of {name.replace('_', ' ')}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]


class AVEDatasetVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="AVEDatasetVideoZeroShot",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from video only.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVE-Dataset",
            "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-09-01"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name: str = "video"
    label_column_name: str = "label"

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a video of {name.replace('_', ' ')}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
