from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MNISTClassification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="MNIST",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "ylecun/mnist",
            "revision": "77f3279092a1c1579b2250db8eafed0ad422088c",
        },
        type="ImageClassification",
        category="i2c",
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
    )
