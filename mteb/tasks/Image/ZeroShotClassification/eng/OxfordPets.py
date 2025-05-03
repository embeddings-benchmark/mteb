from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordPetsZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="OxfordPetsZeroShot",
        description="Classifying animal images.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "isaacchung/OxfordPets",
            "revision": "557b480fae8d69247be74d9503b378a09425096f",
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
        descriptive_stats={
            "n_samples": {"test": 3669},
            "avg_character_length": {"test": 431.4},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}, a type of pet."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
