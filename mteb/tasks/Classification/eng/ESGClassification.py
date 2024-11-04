from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESGClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ESGClassification",
        description="A finance dataset performs sentence classification under the environmental, social, and corporate governance (ESG) framework.",
        # reference="",
        dataset={
            "path": "FinanceMTEB/ESG",
            "revision": "521d56feabadda80b11d6adcc6b335d4c5ad8285",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
    )
