from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TheGoldmanEnRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TheGoldmanEnRetrieval",
        description="Goldman Sachs Financial Dictionary.",
        reference="",
        dataset={
            "path": "FinanceMTEB/TheGoldmanEncyclopedia-en",
            "revision": "c33ce9ced2224b9c6d9e425196edc36f53400ec1",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
