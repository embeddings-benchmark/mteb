from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class OxfordPetsZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OxfordPetsZeroShot",
        description="Classifying animal images.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "mteb/OxfordPetsZeroShot",
            "revision": "ad0f86f308c342b4284ffcd96f0e5255990f8b75",
        },
        type="ZeroShotClassification",
        category="i2t",
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
        modalities=["text", "image"],
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
