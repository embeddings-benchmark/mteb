from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TheGoldmanZhRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TheGoldmanZhRetrieval",
        description="Goldman Sachs Financial Dictionary.",
        reference="",
        dataset={
            "path": "FinanceMTEB/TheGoldmanEncyclopedia-zh",
            "revision": "09cf73149a1e1a81b32ab9e968cdfef9a7a4a1a5",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
