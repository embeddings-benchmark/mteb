from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@article{smaira2020short,
  author = {Smaira, Lucas and Carreira, Joao and Noland, Eric and Clancy, Ellen and Wu, Amy and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:2010.10864},
  title = {A Short Note on the Kinetics-700-2020 Human Action Dataset},
  year = {2020},
}
"""


class Kinetics700VAZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Kinetics700VAZeroShot",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities.",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/kinetics-700-2020",
            "revision": "e9f50aa09759e014b8afc16cc27ec536d4c0747f",
        },
        type="VideoZeroshotClassification",
        category="va2t",
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

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]


class Kinetics700VZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Kinetics700VZeroShot",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses video only.",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/kinetics-700-2020",
            "revision": "e9f50aa09759e014b8afc16cc27ec536d4c0747f",
        },
        type="VideoZeroshotClassification",
        category="v2t",
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

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]
