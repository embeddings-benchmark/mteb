from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class DISCFinLLMComputingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DISCFinLLMComputingRetrieval",
        description="Financial scenario QA dataset incuding retrieval task.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-Computing",
            "revision": "2342751577b08c5ee989174fdac8f08d6d7f3e88",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
