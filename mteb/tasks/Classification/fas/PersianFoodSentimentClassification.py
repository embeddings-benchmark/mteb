from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PersianFoodSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersianFoodSentimentClassification",
        description="Persian Food Review Dataset",
        reference="https://hooshvare.github.io/docs/datasets/sa",
        dataset={
            "path": "asparius/Persian-Food-Sentiment",
            "revision": "92ba517dfd22f6334111ad84154d16a2890f5b1d",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2020-01-01", "2020-05-31"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @article{ParsBERT,
            title={ParsBERT: Transformer-based Model for Persian Language Understanding},
            author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
            journal={ArXiv},
            year={2020},
            volume={abs/2005.12515}
        }
        """,
        n_samples={"train": 56700, "validation": 6300, "test": 7000},
        avg_character_length={"train": 89.61, "validation": 90.37, "test": 90.58},
    )
