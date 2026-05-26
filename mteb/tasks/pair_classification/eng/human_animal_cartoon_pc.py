from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@inproceedings{dong2023simmmdg,
  author = {Dong, Hao and Nejjar, Ismail and Sun, Han and Chatzi, Eleni and Fink, Olga},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Sim{MMDG}: A Simple and Effective Framework for Multi-modal Domain Generalization},
  year = {2023},
}
"""

DESC_BASE = (
    "Pair classification on the Human-Animal-Cartoon (HAC) dataset: "
    "determining whether two clips depict the same action (one of seven action "
    "classes) across different actor domains (human, animal, cartoon). "
    "Same-action / different-action pairs are sampled with a fixed seed."
)


class HumanAnimalCartoonVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonVPairClassification",
        description=DESC_BASE + " Uses video only.",
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "zachz/Human-Animal-Cartoon-PC-V",
            "revision": "780fcc3def5d1adf2c2cb8bfab0c618962addb71",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-10-30", "2023-10-30"),
        domains=["Entertainment", "Scene", "Web"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"


class HumanAnimalCartoonVAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonVAPairClassification",
        description=DESC_BASE + " Uses synchronized video and audio.",
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "zachz/Human-Animal-Cartoon-PC-VA",
            "revision": "3ac2ac39f900b5fdf9a7b4b828ed12e24d1121e7",
        },
        type="VideoPairClassification",
        category="va2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-10-30", "2023-10-30"),
        domains=["Entertainment", "Scene", "Web"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )

    input1_column_name = (("video1", "video"), ("audio1", "audio"))
    input2_column_name = (("video2", "video"), ("audio2", "audio"))
    label_column_name: str = "label"
