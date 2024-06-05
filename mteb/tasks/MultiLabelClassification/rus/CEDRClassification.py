from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskMultilabelClassification


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
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("1999-01-01", "2021-09-01"),
        form=["written"],
        domains=["Web", "Social", "Blog"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
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
        n_samples={"test": 1882},
        avg_character_length={"test": 91.2},
    )
