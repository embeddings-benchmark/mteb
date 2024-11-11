from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HeadlinePDDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlinePDDPairClassification",
        description="Financial text sentiment categorization dataset.",
        reference="https://arxiv.org/abs/2009.04202",
        dataset={
            "path": "FinanceMTEB/HeadlinePDD-PairClassification",
            "revision": "ad0150ed63940e88846659e46be5138aa0db6c85",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
    )

    sentence_1_column = "sent1"
    sentence_2_column = "sent2"
