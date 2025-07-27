from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class DTDZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="DTDZeroShot",
        description="Describable Textures Dataset in 47 categories.",
        reference="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        dataset={
            "path": "tanganke/dtd",
            "revision": "d2afa97d9f335b1a6b3b09c637aef667f98f966e",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2014-01-01",
            "2014-03-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Textures recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{cimpoi14describing,
  author = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
  booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
  title = {Describing Textures in the Wild},
  year = {2014},
}
""",
        descriptive_stats={
            "n_samples": {"test": 1880},
            "avg_character_length": {"test": 456},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of {name} texture."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
