from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FGVCAircraftPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FGVCAircraftPairClassification",
        description="Classifying image pairs as showing the same or different aircraft model variant, constructed from the FGVC-Aircraft test split.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "shriyasudhakar/FGVCAircraftPairClassification",
            "revision": "11a2b6646cdd236bf3e979ea4d83bb78260bccae",
        },
        type="ImagePairClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2009-01-01", "2010-01-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""@misc{maji2013finegrainedvisualclassificationaircraft,
  archiveprefix = {arXiv},
  author = {Subhransu Maji and Esa Rahtu and Juho Kannala and Matthew Blaschko and Andrea Vedaldi},
  eprint = {1306.5151},
  primaryclass = {cs.CV},
  title = {Fine-Grained Visual Classification of Aircraft},
  url = {https://arxiv.org/abs/1306.5151},
  year = {2013},
}""",
    )

    input1_column_name: str = "image1"
    input2_column_name: str = "image2"
    label_column_name: str = "label"
