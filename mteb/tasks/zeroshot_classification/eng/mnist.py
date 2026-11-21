from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MNISTZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MNISTZeroShot",
        description="Classifying handwritten digits.",
        reference="https://en.wikipedia.org/wiki/MNIST_database",
        dataset={
            "path": "mteb/MNISTZeroShot",
            "revision": "626ff684f46fd8fe0a6646c387f2d23455d9cea7",
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
