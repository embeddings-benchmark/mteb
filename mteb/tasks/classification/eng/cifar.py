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
            "path": "uoft-cs/cifar10",
            "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
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
            "path": "uoft-cs/cifar100",
            "revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
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
