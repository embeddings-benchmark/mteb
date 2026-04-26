from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MusicAVQACLSClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MusicAVQACLSClassification",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),  # around time of conference
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{li2022learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {19108--19118},
  title = {Learning to answer questions in dynamic audio-visual scenarios},
  year = {2022},
}
""",
        is_beta=True,
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "label"
    is_cross_validation: bool = True
    train_split: str = "test"


class MusicAVQACLSVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MusicAVQACLSVideoClassification",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type.",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),  # around time of conference
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{li2022learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {19108--19118},
  title = {Learning to answer questions in dynamic audio-visual scenarios},
  year = {2022},
}
""",
        is_beta=True,
    )
    input_column_name = "video"
    label_column_name: str = "label"
    is_cross_validation: bool = True
    train_split: str = "test"
