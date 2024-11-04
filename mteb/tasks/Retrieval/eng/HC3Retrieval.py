from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class HC3Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3Retrieval",
        description="A human-ChatGPT comparison finance corpus",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/HPC3-finance",
            "revision": "7018353fb281b866e5934eeb496251be4ad3585f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
