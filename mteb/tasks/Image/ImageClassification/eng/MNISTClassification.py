from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks import AbsTaskImageClassification


class MNISTClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="MNIST",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "ylecun/mnist",
            "revision": "b06aab39e05f7bcd9635d18ed25d06eae523c574",
            "trust_remote_code": True,
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2010-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@article{lecun2010mnist,
        title={MNIST handwritten digit database},
        author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
        journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
        volume={2},
        year={2010}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )
