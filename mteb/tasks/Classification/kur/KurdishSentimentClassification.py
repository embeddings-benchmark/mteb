from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KurdishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KurdishSentimentClassification",
        description="Kurdish Sentiment Dataset",
        reference="https://link.springer.com/article/10.1007/s10579-023-09716-6",
        dataset={
            "path": "asparius/Kurdish-Sentiment",
            "revision": "f334d90a9f68cc3af78cc2a2ece6a3b69408124c",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kur-Arab"],
        main_score="accuracy",
        date=("2023-01-01", "2024-01-02"),
        domains=["Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY 4.0",
        annotations_creators="derived",
        dialect=["Sorani"],
        sample_creation="found",
        bibtex_citation="""
        @article{article,
            author = {Badawi, Soran and Kazemi, Arefeh and Rezaie, Vali},
            year = {2024},
            month = {01},
            pages = {1-20},
            title = {KurdiSent: a corpus for kurdish sentiment analysis},
            journal = {Language Resources and Evaluation},
            doi = {10.1007/s10579-023-09716-6}
            }
        """,
        descriptive_stats={
            "n_samples": {"train": 6000, "test": 1987},
            "avg_character_length": {"train": 59.38, "test": 56.11},
        },
    )
