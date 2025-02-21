from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import AbsTaskAudioClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSD50KClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="FSD50K",
        description="Multilabel Audio Classification.",
        reference="https://huggingface.co/datasets/Fhrozen/FSD50k",
        dataset={
            "path": "Fhrozen/FSD50k",
            "revision": "67e4d8c2570caef0f90d48fdb756b337875d91db",
        },
        type="AudioClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",  # restart from here..
        date=(
            "2008-01-01",
            "2009-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=""" @TECHREPORT{Krizhevsky09learningmultiple,
            author = {Alex Krizhevsky},
            title = {Learning multiple layers of features from tiny images},
            institution = {},
            year = {2009}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )
    image_column_name: str = "img"

