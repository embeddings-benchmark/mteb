from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2800


class SlovakSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakSentimentClassification",
        description="Slovak Sentiment Classification Dataset",
        reference="https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment",
        dataset={
            "path": "sepidmnorozy/Slovak_sentiment",
            "revision": "e698d1df52766d73ae1cc569dfc622527c329a08",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slk-Cyrs"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        """,
        n_samples={"train": N_SAMPLES, "validation": 522, "test": 1040},
        avg_character_length={"train": 87.96, "validation": 84.96, "test": 91.95},
    )
