from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MNISTZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MNISTZeroShot",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "ylecun/mnist",
            "revision": "77f3279092a1c1579b2250db8eafed0ad422088c",
        },
        type="ZeroShotClassification",
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
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
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

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of the number: '{name}'."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
