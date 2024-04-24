from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpanishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpanishSentimentClassification",
        description="A Spanish dataset for sentiment classification.",
        reference="https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment",
        dataset={
            "path": "sepidmnorozy/Spanish_sentiment",
            "revision": "2a6e340e4b59b7c0a78c03a0b79ac27e1b4a2662",
        },
        type="Classification",
        category="s2s",
        date=("2022-08-16", "2022-08-16"),
        eval_splits=["validation", "test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"validation": 147, "test": 296},
        avg_character_length={"validation": 85.02, "test": 87.91},
    )
