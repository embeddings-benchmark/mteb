from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AFQMCPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AFQMCPairClassification",
        description="Ant Financial Question Matching Corpus.",
        reference="https://tianchi.aliyun.com/dataset/106411",
        dataset={
            "path": "FinanceMTEB/AFQMC-PairClassification",
            "revision": "623887e33b741cf9e5faa2bae12a4269c1de8fec",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="max_ap",
    )

    sentence_1_column = "sent1"
    sentence_2_column = "sent2"
