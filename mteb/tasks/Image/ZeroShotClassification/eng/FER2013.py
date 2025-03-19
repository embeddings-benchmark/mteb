from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class FER2013ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="FER2013ZeroShot",
        description="Classifying facial emotions.",
        reference="https://arxiv.org/abs/1412.6572",
        dataset={
            "path": "clip-benchmark/wds_fer2013",
            "revision": "9399b94167523fe5c40b3a857e24ef931ee4395b",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2014-01-01",
            "2014-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Emotion recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@misc{goodfellow2015explainingharnessingadversarialexamples,
        title={Explaining and Harnessing Adversarial Examples},
        author={Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
        year={2015},
        eprint={1412.6572},
        archivePrefix={arXiv},
        primaryClass={stat.ML},
        url={https://arxiv.org/abs/1412.6572},
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 7178},
            "avg_character_length": {"test": 431.4},
        },
    )
    image_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        return [f"a photo of a {name} looking face." for name in labels]
