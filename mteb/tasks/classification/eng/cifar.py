from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CIFAR10Classification(AbsTaskClassification):
    input_column_name: str = "img"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="CIFAR10",
        description="Classifying images from 10 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar10",
        dataset={
            "path": "mteb/cifar10",
            "revision": "69a62dd171e24a133d193073e73fda4dbb823266",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
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


class CIFAR100Classification(AbsTaskClassification):
    input_column_name: str = "img"
    label_column_name: str = "fine_label"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="CIFAR100",
        description="Classifying images from 100 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar100",
        dataset={
            "path": "mteb/cifar100",
            "revision": "ac5511f885f65fb75d31a9d8810a8440913f7721",
        },
        type="ImageClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
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
