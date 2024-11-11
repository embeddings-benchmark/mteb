from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TATQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TATQARetrieval",
        description="A question answering benchmark on a Hybrid of tabular and textual content in finance",
        reference="https://arxiv.org/pdf/2105.07624",
        dataset={
            "path": "FinanceMTEB/TATQA",
            "revision": "11b15221dd850044dc2261ce633e692851c8b7e2",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
