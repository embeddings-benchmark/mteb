from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SwedishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwedishSentimentClassification",
        description="Dataset of Swedish reviews scarped from various public available websites",
        reference="https://huggingface.co/datasets/swedish_reviews",
        dataset={
            "path": "mteb/SwedishSentimentClassification",
            "revision": "39e35f55d58338ebd602f8d83b52cfe027f5146a",
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
    )
