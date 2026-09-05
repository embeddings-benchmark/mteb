from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FER2013PairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FER2013PairClassification",
        description="Classifying face image pairs as expressing the same or different emotion, constructed from the FER2013 test split.",
        reference="https://arxiv.org/abs/1412.6572",
        dataset={
            "path": "shriyasudhakar/FER2013PairClassification",
            "revision": "ce5bdc0fdd9717dd98b40b4356688ea076d150e0",
        },
        type="ImagePairClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-01-01", "2014-12-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""@misc{goodfellow2015explainingharnessingadversarialexamples,
  archiveprefix = {arXiv},
  author = {Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
  eprint = {1412.6572},
  primaryclass = {stat.ML},
  title = {Explaining and Harnessing Adversarial Examples},
  url = {https://arxiv.org/abs/1412.6572},
  year = {2015},
}""",
    )

    input1_column_name: str = "image1"
    input2_column_name: str = "image2"
    label_column_name: str = "label"
