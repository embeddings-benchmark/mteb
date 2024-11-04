from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Apple10KRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Apple10KRetrieval",
        description="A RAG benchmark for finance applications.",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/Apple-10K-2022",
            "revision": "27d0b84029e5de607fc7a8e2fb2a315e9b71f570",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
