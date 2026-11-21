from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CIFAR10ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CIFAR10ZeroShot",
        description="Classifying images from 10 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar10",
        dataset={
            "path": "mteb/CIFAR10ZeroShot",
            "revision": "a56fb36de8a8d9d81dcf0f16554c3cb36f44e68b",
        },
        type="ZeroShotClassification",
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
        modalities=["text", "image"],
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


class CIFAR100ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CIFAR100ZeroShot",
        description="Classifying images from 100 classes.",
        reference="https://huggingface.co/datasets/uoft-cs/cifar100",
        dataset={
            "path": "mteb/CIFAR100ZeroShot",
            "revision": "1c66c783863e322bd0590a5858e48dfdde6fbcec",
        },
        type="ZeroShotClassification",
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
        modalities=["text", "image"],
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
