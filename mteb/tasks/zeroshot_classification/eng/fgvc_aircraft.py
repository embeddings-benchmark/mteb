from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class FGVCAircraftZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="FGVCAircraftZeroShot",
        description="Classifying aircraft images from 41 manufacturers and 102 variants.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "mteb/FGVCAircraftZeroShot",
            "revision": "2d33d00400099b87bfec8b04f2876fb10ca07487",
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
    label_column_name: str = "variant"  # could be family, manufacturer, or variant. Variant has the higher number of classes.

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}, a type of aircraft."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
