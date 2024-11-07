from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinNSPClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinNSPClassification",
        description="Financial negative news and its subject determination dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinNSP",
            "revision": "1d3ae2b90b692ca702a76f26b94c7cb09b23ca14",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        if "sentence" in self.dataset:
            self.dataset = self.dataset.rename_column("sentence", "text")
