from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@article{carreira2018short,
  author = {Carreira, Joao and Noland, Eric and Banki-Horvath, Andras and Hillier, Chloe and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:1808.01340},
  title = {A Short Note about Kinetics-600},
  year = {2018},
}
"""


class Kinetics600VAZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics600VAZeroShot",
        description="Kinetics-600 is a large-scale action recognition dataset containing 600 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities. Sampled the official test split, capped at ~16 per class across 600 classes (~9,576 examples).",
        reference="https://arxiv.org/abs/1808.01340",
        dataset={
            "path": "mteb/Kinetics600VAZeroShot",
            "revision": "f48f629978bb4dff56b1373169aae00c4c1a3a2b",
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
        domains=["Activity", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class Kinetics600VZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics600VZeroShot",
        description="Kinetics-600 is a large-scale action recognition dataset containing 600 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses video only. Sampled the official test split, capped at ~16 per class across 600 classes (~9,576 examples).",
        reference="https://arxiv.org/abs/1808.01340",
        dataset={
            "path": "mteb/Kinetics600VZeroShot",
            "revision": "fd4e4adfc646cbfc6c1c86459076968314413909",
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
        domains=["Activity", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
