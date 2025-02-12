from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinEvaEncyclopediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinEvaEncyclopediaRetrieval",
        description="Financial scenario QA dataset provides terminology used in the financial industry.",
        reference="https://github.com/alipay/financial_evaluation_dataset",
        dataset={
            "path": "FinanceMTEB/FinEvaEncyclopedia",
            "revision": "3f4e6b66d58ba1514718e3823f0cb818609dce25",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
