from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TradeTheEventNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TradeTheEventNewsRetrieval",
        description="A dataset comprising finance news articles, each paired with its corresponding headline and stock ticker symbol.",
        reference="https://aclanthology.org/2021.findings-acl.186.pdf",
        dataset={
            "path": "FinanceMTEB/TradeTheEventNews",
            "revision": "9d499d833355b8ef1810073e5f2aa174d743d4d0",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
