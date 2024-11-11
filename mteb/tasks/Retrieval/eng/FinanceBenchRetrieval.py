from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinanceBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBenchRetrieval",
        description="Open book financial question answering (QA)",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "FinanceMTEB/FinanceBench",
            "revision": "4738010357e3dda4b337abbde86d5b36c3118c8f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
