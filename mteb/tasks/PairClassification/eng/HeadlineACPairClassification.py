from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HeadlineACPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlineACPairClassification",
        description="Financial text sentiment categorization dataset.",
        reference="",
        dataset={
            "path": "FinanceMTEB/HeadlinePDU-PairClassification",
            "revision": "afa5a612a01b9c0a8058fba44a3bdf66173583eb",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ap",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
