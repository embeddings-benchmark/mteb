from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class MNISTZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="MNISTZeroShot",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "mteb/mnist",
            "revision": "cf6afcbae72cc3bb1fc07a9480fe0d8e0b615fcd",
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
        bibtex_citation=r"""
@article{lecun1998gradient,
  author = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal = {Proceedings of the IEEE},
  number = {11},
  pages = {2278--2324},
  publisher = {Ieee},
  title = {Gradient-based learning applied to document recognition},
  volume = {86},
  year = {1998},
}
""",
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of the number: '{name}'."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
