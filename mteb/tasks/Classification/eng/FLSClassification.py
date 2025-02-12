from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FLSClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FLSClassification",
        dataset={
            "path": "FinanceMTEB/FLS",
            "revision": "39b6719f1d7197df4498fea9fce20d4ad782a083",
        },
        description="A finance dataset detects whether the sentence is a forward-looking statement.",
        reference="https://arxiv.org/abs/2309.13064",
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
        annotations_creators="derived",
        descriptive_stats={
            "num_samples": {"test": 1000},
            "average_text_length": {"test": 187.923},
            "unique_labels": {"test": 3},
            "labels": {
                "test": {
                    "2": {"count": 292},
                    "1": {"count": 539},
                    "0": {"count": 169},
                }
            },
        },
    )
