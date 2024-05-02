from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering
from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class EightTagsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="EightTagsClustering",
        description="Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, "
        "food, medicine, motorization, work, sport and technology.",
        reference="https://aclanthology.org/2020.lrec-1.207.pdf",
        dataset={
            "path": "PL-MTEB/8tags-clustering",
            "revision": "78b962b130c6690659c65abf67bf1c2f030606b6",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )


class PlscClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PlscClusteringS2S",
        description="Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either "
        "on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-clustering-s2s",
            "revision": "39bcadbac6b1eddad7c1a0a176119ce58060289a",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 17534},
        avg_character_length={"test": 84.34},
    )


class PlscClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PlscClusteringP2P",
        description="Clustering of Polish article titles+abstracts from Library of Science "
        "(https://bibliotekanauki.pl/), either on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-clustering-p2p",
            "revision": "8436dd4c05222778013d6642ee2f3fa1722bca9b",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 17537},
        avg_character_length={"test": 1023.21},
    )
