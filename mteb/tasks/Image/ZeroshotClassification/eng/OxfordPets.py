from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroshotClassification import (
    AbsTaskZeroshotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordPetsClassification(AbsTaskZeroshotClassification):
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
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@misc{maji2013finegrainedvisualclassificationaircraft,
            title={Fine-Grained Visual Classification of Aircraft}, 
            author={Subhransu Maji and Esa Rahtu and Juho Kannala and Matthew Blaschko and Andrea Vedaldi},
            year={2013},
            eprint={1306.5151},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/1306.5151}, 
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
