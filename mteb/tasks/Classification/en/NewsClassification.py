from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class NewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NewsClassification",
        description="Large News Classification Dataset",
        dataset={
            "path": "ag_news",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        },
        reference="https://arxiv.org/abs/1509.01626",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        date=None,
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="Apache 2.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=["en-US", "en-GB", "en-AU"],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 7600},
        avg_character_length={"test": 235.29},
    )
