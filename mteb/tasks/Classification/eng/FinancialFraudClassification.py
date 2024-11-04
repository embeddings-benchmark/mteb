from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinancialFraudClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialFraudClassification",
        description="This dataset was used for research in detecting financial fraud.",
        reference="https://github.com/amitkedia007/Financial-Fraud-Detection-Using-LLMs",
        dataset={
            "path": "FinanceMTEB/FinancialFraud",
            "revision": "e569a69e058ad8504f03556cd05c36700767d193",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
