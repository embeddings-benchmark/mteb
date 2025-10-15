from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FGVCAircraftClassification(AbsTaskClassification):
    input_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5
    # could be family, manufacturer, or variant. Variant has the higher number of classes.
    label_column_name: str = "variant"

    metadata = TaskMetadata(
        name="FGVCAircraft",
        description="Classifying aircraft images from 41 manufacturers and 102 variants.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "mteb/FGVCAircraft",
            "revision": "3ae0e4e94e37c610c26df758ec3255fa90d80e67",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2009-01-01",
            "2010-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{maji2013finegrainedvisualclassificationaircraft,
  archiveprefix = {arXiv},
  author = {Subhransu Maji and Esa Rahtu and Juho Kannala and Matthew Blaschko and Andrea Vedaldi},
  eprint = {1306.5151},
  primaryclass = {cs.CV},
  title = {Fine-Grained Visual Classification of Aircraft},
  url = {https://arxiv.org/abs/1306.5151},
  year = {2013},
}
""",
    )
