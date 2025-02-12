from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FOMCClassification",
        dataset={
            "path": "FinanceMTEB/FOMC",
            "revision": "cdaf1306a24bc5e7441c7c871343efdf4c721bc2",
        },
        description="A task of hawkish-dovish classification in finance domain.",
        reference="https://github.com/gtfintechlab/fomc-hawkish-dovish",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2022-10-15"),
        domains=["Finance"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        descriptive_stats={
            "num_samples": {"test": 1000},
            "average_text_length": {"test": 199.403},
            "unique_labels": {"test": 3},
            "labels": {
                "test": {
                    "1": {"count": 263},
                    "2": {"count": 466},
                    "0": {"count": 271},
                }
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
