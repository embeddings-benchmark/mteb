from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Weibo21Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Weibo21Classification",
        description="Fake news detection in finance domain.",
        reference="https://dl.acm.org/doi/pdf/10.1145/3459637.3482139",
        dataset={
            "path": "FinanceMTEB/MDFEND-Weibo21",
            "revision": "db799d3d74bc752cb30b264a6254ab52471f693d",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
