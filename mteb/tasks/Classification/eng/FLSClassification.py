from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FLSClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FLSClassification",
        description="A finance dataset detects whether the sentence is a forward-looking statement.",
        reference="https://arxiv.org/abs/2309.13064",
        dataset={
            "path": "FinanceMTEB/FLS",
            "revision": "39b6719f1d7197df4498fea9fce20d4ad782a083",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )
