from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@article{smaira2020short,
  author = {Smaira, Lucas and Carreira, Joao and Noland, Eric and Clancy, Ellen and Wu, Amy and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:2010.10864},
  title = {A Short Note on the Kinetics-700-2020 Human Action Dataset},
  year = {2020},
}
"""


class Kinetics700VAZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics700VAZeroShot",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities. Sampled the official test split, capped at ~16 per class across 700 classes (~11,190 examples).",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/Kinetics700VAZeroShot",
            "revision": "5bf815d041d9dfecddb5787646288d24fa930d7f",
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


class Kinetics700VZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics700VZeroShot",
        description="Kinetics-700-2020 is a large-scale action recognition dataset containing 700 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses video only. Sampled the official test split, capped at ~16 per class across 700 classes (~11,190 examples).",
        reference="https://arxiv.org/abs/2010.10864",
        dataset={
            "path": "mteb/Kinetics700VZeroShot",
            "revision": "a7e4e0e7dac93b72e9948230ac19c9cd9593e1f6",
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
