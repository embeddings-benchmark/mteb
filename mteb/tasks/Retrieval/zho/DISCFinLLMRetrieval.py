from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class DISCFinLLMRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DISCFinLLMRetrieval",
        description="Financial scenario QA dataset incuding retrieval task.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-Retrieval",
            "revision": "00fb9b68b7e204b1fd03a3433ad81cbc6282aa0c",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
