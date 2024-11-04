from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SmoothNLPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SmoothNLPRetrieval",
        description="Chinese finance news dataset.",
        reference="https://github.com/smoothnlp/FinancialDatasets/",
        dataset={
            "path": "FinanceMTEB/SmoothNLPNews",
            "revision": "2d2476d29f78ed46e818b00ba589d7d756839595",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
