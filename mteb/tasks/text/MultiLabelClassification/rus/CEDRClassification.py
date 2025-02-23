from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_multilabel_classification import (
    AbsTextMultilabelClassification,
)


class CEDRClassification(AbsTextMultilabelClassification):
    metadata = TaskMetadata(
        name="CEDRClassification",
        dataset={
            "path": "ai-forever/cedr-classification",
            "revision": "c0ba03d058e3e1b2f3fd20518875a4563dd12db4",
        },
        description="Classification of sentences by emotions, labeled into 5 categories (joy, sadness, surprise, fear, and anger).",
        reference="https://www.sciencedirect.com/science/article/pii/S1877050921013247",
        type="MultilabelClassification",
        category="t2t",
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
        prompt="Given a comment as query, find expressed emotions (joy, sadness, surprise, fear, and anger)",
    )
