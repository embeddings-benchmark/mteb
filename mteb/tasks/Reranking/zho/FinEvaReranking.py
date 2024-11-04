from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinEvaReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="FinEvaReranking",
        description="Financial scenario QA dataset",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaRetrieval-reranking",
            "revision": "20a5ca0e51f8948f59dd79ee648e9d7f2f76f25f",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map",
    )
