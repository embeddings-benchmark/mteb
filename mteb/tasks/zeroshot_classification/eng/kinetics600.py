from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@article{carreira2018short,
  author = {Carreira, Joao and Noland, Eric and Banki-Horvath, Andras and Hillier, Chloe and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:1808.01340},
  title = {A Short Note about Kinetics-600},
  year = {2018},
}
"""


class Kinetics600VAZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Kinetics600VAZeroShot",
        description="Kinetics-600 is a large-scale action recognition dataset containing 600 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities.",
        reference="https://arxiv.org/abs/1808.01340",
        dataset={
            "path": "mteb/kinetics-600",
            "revision": "a7be893c873e39341a96753e99bfd7b7025aaaf9",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2018-08-03",
            "2018-08-03",
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


class Kinetics600VZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Kinetics600VZeroShot",
        description="Kinetics-600 is a large-scale action recognition dataset containing 600 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses video only.",
        reference="https://arxiv.org/abs/1808.01340",
        dataset={
            "path": "mteb/kinetics-600",
            "revision": "a7be893c873e39341a96753e99bfd7b7025aaaf9",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2018-08-03",
            "2018-08-03",
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