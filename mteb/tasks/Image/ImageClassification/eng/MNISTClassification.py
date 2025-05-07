from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MNISTClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="MNIST",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "ylecun/mnist",
            "revision": "77f3279092a1c1579b2250db8eafed0ad422088c",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2010-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{lecun2010mnist,
  author = {LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal = {ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  title = {MNIST handwritten digit database},
  volume = {2},
  year = {2010},
}
""",
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )
