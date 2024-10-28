from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 1024


class SwedishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwedishSentimentClassification",
        description="Dataset of Swedish reviews scarped from various public available websites",
        reference="https://huggingface.co/datasets/swedish_reviews",
        dataset={
            "path": "timpal0l/swedish_reviews",
            "revision": "105ba6b3cb99b9fd64880215be469d60ebf44a1b",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"validation": N_SAMPLES, "test": N_SAMPLES},
            "avg_character_length": {"validation": 499.3, "test": 498.1},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
