from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClustering import AbsTaskImageClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class CIFAR10Clustering(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="CIFAR10Clustering",
        description="Clustering images from 10 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar10",
        dataset={
            "path": "uoft-cs/cifar10",
            "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
        },
        type="ImageClustering",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=(
            "2008-01-01",
            "2009-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@techreport{Krizhevsky09learningmultiple,
  author = {Alex Krizhevsky},
  institution = {},
  title = {Learning multiple layers of features from tiny images},
  year = {2009},
}
""",
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )

    image_column_name: str = "img"


class CIFAR100Clustering(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="CIFAR100Clustering",
        description="Clustering images from 100 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar100",
        dataset={
            "path": "uoft-cs/cifar100",
            "revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
        },
        type="ImageClustering",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=(
            "2008-01-01",
            "2009-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@techreport{Krizhevsky09learningmultiple,
  author = {Alex Krizhevsky},
  institution = {},
  title = {Learning multiple layers of features from tiny images},
  year = {2009},
}
""",
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )
    image_column_name: str = "img"
    label_column_name: str = "fine_label"
