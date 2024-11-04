from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SemEva2017Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SemEva2017Classification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://alt.qcri.org/semeval2017/task5/",
        dataset={
            "path": "FinanceMTEB/SemEva2017_Headline",
            "revision": "f0e198ba04c23d949ef803ce32ee1e4f2d8d3696",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
