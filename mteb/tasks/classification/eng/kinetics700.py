from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@article{smaira2020short,
  author = {Smaira, Lucas and Carreira, Joao and Noland, Eric and Clancy, Ellen and Wu, Amy and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:2010.10864},
  title = {A Short Note on the Kinetics-700-2020 Human Action Dataset},
  year = {2020},
}
"""


class Kinetics700VAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kinetics700VA",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities.",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/kinetics-700-2020",
            "revision": "e9f50aa09759e014b8afc16cc27ec536d4c0747f",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-10-21",
            "2020-10-21",
        ),
        domains=["Web", "Scene"],
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


class Kinetics700VClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kinetics700V",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses video only.",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/kinetics-700-2020",
            "revision": "e9f50aa09759e014b8afc16cc27ec536d4c0747f",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-10-21",
            "2020-10-21",
        ),
        domains=["Web", "Scene"],
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
