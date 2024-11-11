from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FOMCClassification",
        description="A task of hawkish-dovish classification in finance domain.",
        reference="https://github.com/gtfintechlab/fomc-hawkish-dovish",
        dataset={
            "path": "FinanceMTEB/FOMC",
            "revision": "cdaf1306a24bc5e7441c7c871343efdf4c721bc2",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
