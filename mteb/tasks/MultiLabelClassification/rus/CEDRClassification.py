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
        bibtex_citation=r"""
@article{sboev2021data,
  author = {Sboev, Alexander and Naumov, Aleksandr and Rybka, Roman},
  journal = {Procedia Computer Science},
  pages = {637--642},
  publisher = {Elsevier},
  title = {Data-Driven Model for Emotion Detection in Russian Texts},
  volume = {190},
  year = {2021},
}
""",
        prompt="Given a comment as query, find expressed emotions (joy, sadness, surprise, fear, and anger)",
    )
