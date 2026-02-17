from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class CIFAR10Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="CIFAR10Clustering",
        description="Clustering images from 10 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar10",
        dataset={
            "path": "mteb/cifar10",
            "revision": "69a62dd171e24a133d193073e73fda4dbb823266",
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
            "path": "mteb/cifar100",
            "revision": "ac5511f885f65fb75d31a9d8810a8440913f7721",
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
