from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class CIFAR10Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="CIFAR10Clustering",
        description="Clustering images from 10 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar10",
        dataset={
            "path": "uoft-cs/cifar10",
            "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
        },
        type="ImageClustering",
        category="i2c",
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
    )

    input_column_name: str = "img"
    label_column_name: str = "label"


class CIFAR100Clustering(AbsTaskClusteringLegacy):
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
    )
    input_column_name: str = "img"
    label_column_name: str = "fine_label"
