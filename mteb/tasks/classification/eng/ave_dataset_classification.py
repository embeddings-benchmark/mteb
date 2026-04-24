from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AVEDatasetClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AVEDatasetClassification",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from synchronized video and audio.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVE-Dataset",
            "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-09-01"),  # some time before conference
        domains=["Web", "AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{tian2018audio,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {Proceedings of the European conference on computer vision (ECCV)},
  pages = {247--263},
  title = {Audio-visual event localization in unconstrained videos},
  year = {2018},
}
""",
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "label"
    is_cross_validation: bool = False
    
    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset["train"] = self.dataset["train"].select(range(2048))


class AVEDatasetVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AVEDatasetVideoClassification",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from video only.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVE-Dataset",
            "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
        },
        type="VideoClassification",
        category="v2c",
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
        bibtex_citation=r"""
@inproceedings{tian2018audio,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {Proceedings of the European conference on computer vision (ECCV)},
  pages = {247--263},
  title = {Audio-visual event localization in unconstrained videos},
  year = {2018},
}
""",
    )
    input_column_name = "video"
    label_column_name: str = "label"
    is_cross_validation: bool = False

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset["train"] = self.dataset["train"].select(range(2048))
