from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata



class GreekSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GreekSentimentClassification",
        dataset={
            "path": "sepidmnorozy/Greek_sentiment",
            "revision": "94b245f3ccdf8e8b2cbf8f13f55eee820b70eccf",
        },
        description="Greek sentiment analysis dataset.",
        reference="https://huggingface.co/datasets/sepidmnorozy/Greek_sentiment",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        date=None,
        form=["written"],
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"validation": 383, "test": 767},
        avg_character_length={"validation": 0, "test": 0},
    )
