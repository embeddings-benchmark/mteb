from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SemEva2017Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SemEva2017Classification",
        dataset={
            "path": "FinanceMTEB/SemEva2017_Headline",
            "revision": "f0e198ba04c23d949ef803ce32ee1e4f2d8d3696",
        },
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://alt.qcri.org/semeval2017/task5/",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-07-01", "2017-12-31"),
        domains=["Finance"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        descriptive_stats={
            "num_samples": {"test": 343},
            "average_text_length": {"test": 59.80466472303207},
            "unique_labels": {"test": 3},
            "labels": {
                "test": {
                    "0": {"count": 122},
                    "2": {"count": 204},
                    "1": {"count": 17},
                }
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
