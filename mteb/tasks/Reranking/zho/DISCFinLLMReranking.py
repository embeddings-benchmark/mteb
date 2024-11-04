from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class DISCFinLLMReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="DISCFinLLMReranking",
        description="Financial scenario QA dataset",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-reranking",
            "revision": "e97ba5a0620442edb312b1f8c0ab4fe04096ed59",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map",
    )
