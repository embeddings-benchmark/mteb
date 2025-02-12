from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinSentClassification",
        dataset={
            "path": "FinanceMTEB/FinSent",
            "revision": "68ee0f0abf596e371ef6a308f685071e3b737bbb",
        },
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://finsent.hkust.edu.hk/",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-09-23", "2023-09-23"),
        domains=["Finance"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived", # the annotations are a mix of derived, LM-generated and reviewed and expert-annotated. but derived is the predominant source.
        descriptive_stats={
            "num_samples": {"test": 1000},
            "average_text_length": {"test": 138.939},
            "unique_labels": {"test": 3},
            "labels": {
                "0": {"test": {"count": 465}},
                "1": {"test": {"count": 358}},
                "2": {"test": {"count": 177}},
            },
        },
    )
