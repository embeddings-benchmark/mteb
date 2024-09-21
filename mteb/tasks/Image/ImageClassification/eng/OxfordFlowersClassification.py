from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordFlowersClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordFlowersClassification",
        description="Classifying flowers",
        reference="https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train",
        dataset={
            "path": "nelorth/oxford-flowers",
            "revision": "a37b1891609c0376fa81eced756e7863e1bd873b",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="found",
        bibtex_citation="""d""",
        descriptive_stats={
            "n_samples": {"test": 400000},
            "avg_character_length": {"test": 431.4},
        },
    )
