from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{dong2023simmmdg,
  title = {Sim{MMDG}: A Simple and Effective Framework for Multi-modal Domain Generalization},
  author = {Dong, Hao and Nejjar, Ismail and Sun, Han and Chatzi, Eleni and Fink, Olga},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023},
}
"""


class HumanAnimalCartoonVAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonVA",
        description=(
            "Human-Animal-Cartoon (HAC) is a multi-domain action recognition "
            "dataset containing clips of humans, animals, and cartoon figures. "
            "This MTEB subset uses video and audio clips labelled with one of "
            "seven actions. This variant uses both video and audio modalities."
        ),
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "mteb/Human-Animal-Cartoon",
            "revision": "d38566c4bb055c7325314d3e46610792c2799c4b",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "action"
    train_split: str = "test"
    is_cross_validation: bool = True


class HumanAnimalCartoonVClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonV",
        description=(
            "Human-Animal-Cartoon (HAC) is a multi-domain action recognition "
            "dataset containing clips of humans, animals, and cartoon figures. "
            "This MTEB subset uses video and audio clips labelled with one of "
            "seven actions. This variant uses only the video modality."
        ),
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "mteb/Human-Animal-Cartoon",
            "revision": "d38566c4bb055c7325314d3e46610792c2799c4b",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = "video"
    label_column_name: str = "action"
    train_split: str = "test"
    is_cross_validation: bool = True
