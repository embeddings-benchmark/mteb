from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinFEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinFEClassification",
        description="Financial social media text sentiment categorization dataset.",
        reference="https://arxiv.org/abs/2302.09432",
        dataset={
            "path": "FinanceMTEB/FinFE",
            "revision": "01034e2afdce0f7fa9a51a03aa0fdc1e3d576b05",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
