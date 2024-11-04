from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TradeTheEventEncyclopediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TradeTheEventEncyclopediaRetrieval",
        description="Financial terms and explanations.",
        reference="https://aclanthology.org/2021.findings-acl.186.pdf",
        dataset={
            "path": "FinanceMTEB/TradeTheEventEncyclopedia",
            "revision": "7fa70ba6624011d65a311df86193b4b5587969bc",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
