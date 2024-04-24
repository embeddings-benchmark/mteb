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
        date=None,
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=None,
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 296},
        avg_character_length=None,
    )
