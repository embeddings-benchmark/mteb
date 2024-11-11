from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinEvaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinEvaRetrieval",
        description="Financial scenario QA dataset incuding retrieval task..",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaRetrieval",
            "revision": "ddd402799f8c6d8c8c7ec662a18641e276cfad31",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
