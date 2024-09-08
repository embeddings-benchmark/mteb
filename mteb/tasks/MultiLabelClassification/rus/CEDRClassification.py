from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class CEDRClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="CEDRClassification",
        dataset={
            "path": "ai-forever/cedr-classification",
            "revision": "c0ba03d058e3e1b2f3fd20518875a4563dd12db4",
        },
        description="Classification of sentences by emotions, labeled into 5 categories (joy, sadness, surprise, fear, and anger).",
        reference="https://www.sciencedirect.com/science/article/pii/S1877050921013247",
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("1999-01-01", "2021-09-01"),
        domains=["Web", "Social", "Blog", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{sboev2021data,
        title={Data-Driven Model for Emotion Detection in Russian Texts},
        author={Sboev, Alexander and Naumov, Aleksandr and Rybka, Roman},
        journal={Procedia Computer Science},
        volume={190},
        pages={637--642},
        year={2021},
        publisher={Elsevier}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 1882},
            "test": {
                "average_text_length": 91.20563230605738,
                "average_label_per_text": 0.620616365568544,
                "num_samples": 1882,
                "unique_labels": 6,
                "labels": {
                    "null": {"count": 734},
                    "3": {"count": 141},
                    "2": {"count": 170},
                    "1": {"count": 379},
                    "0": {"count": 353},
                    "4": {"count": 125},
                },
            },
        },
    )
