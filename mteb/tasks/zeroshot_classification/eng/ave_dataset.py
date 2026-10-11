from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{Tian_2018_ECCV,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month = {September},
  title = {Audio-Visual Event Localization in Unconstrained Videos},
  year = {2018},
}
"""


class AVEDatasetZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVEDatasetZeroShot",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from synchronized video and audio. This variant uses both video and audio modalities. Used the official train and test splits across 28 audio-visual event classes (~3,312 train / ~402 test).",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVEDatasetZeroShot",
            "revision": "00a4288e4e190645e6d60638b919c3b14ddde4c3",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-09-01"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class AVEDatasetVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVEDatasetVideoZeroShot",
        description="Audio-Visual Event (AVE) classification dataset containing 28 event categories such as church bell, truck, dog barking, and other everyday sounds sourced from YouTube videos. The goal is to predict the sound event category from video only. Used the official train and test splits across 28 audio-visual event classes (~3,312 train / ~402 test).",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/AVEDatasetVideoZeroShot",
            "revision": "7ae5311ef340164a87cebec006b2a225e3961b8d",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-09-01"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
