from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)

CITATION = r"""
@inproceedings{li2022learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {19108--19118},
  title = {Learning to answer questions in dynamic audio-visual scenarios},
  year = {2022},
}
"""


class MusicAVQACLSAudioVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MusicAVQACLSAudioVideoZeroShot",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type. This zero-shot variant uses both video and audio modalities.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
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
            f"a video of someone playing {name.replace('_', ' ')}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]


class MusicAVQACLSVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MusicAVQACLSVideoZeroShot",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type. This zero-shot variant uses video only.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name: str = "video"
    label_column_name: str = "label"

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a video of someone playing {name.replace('_', ' ')}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
