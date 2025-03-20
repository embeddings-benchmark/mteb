from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FGVCAircraftZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="FGVCAircraftZeroShot",
        description="Classifying aircraft images from 41 manufacturers and 102 variants.",
        reference="https://arxiv.org/abs/1306.5151",
        dataset={
            "path": "HuggingFaceM4/FGVC-Aircraft",
            "revision": "91860adfc9a09aabca5cddb5247442109b38e213",
            "trust_remote_code": True,
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
            "n_samples": {"test": 3333},
            "avg_character_length": {"test": 431.4},
        },
    )
    label_column_name: str = "variant"  ## could be family, manufacturer, or variant. Variant has the higher number of classes.

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}, a type of aircraft."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
