from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinQARetrieval",
        description="Finance numerical reasoning with structured and unstructured evidence",
        reference="https://arxiv.org/pdf/2109.00122",
        dataset={
            "path": "FinanceMTEB/FinQA",
            "revision": "1d4500eefb223c2977187649928561e14108042a",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
