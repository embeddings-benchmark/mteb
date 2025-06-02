from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class Food101ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Food101ZeroShot",
        description="Classifying food.",
        reference="https://huggingface.co/datasets/ethz/food101",
        dataset={
            "path": "ethz/food101",
            "revision": "e06acf2a88084f04bce4d4a525165d68e0a36c38",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bossard14,
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  year = {2014},
}
""",
        descriptive_stats={
            "n_samples": {"validation": 25300},
            "avg_character_length": {"validation": 431.4},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of {name}, a type of food."
            for name in self.dataset["validation"]
            .features[self.label_column_name]
            .names
        ]
