from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinTruthQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinTruthQARetrieval",
        description="Evaluating the Quality of Financial Information Disclosure.",
        reference="https://arxiv.org/pdf/2406.12009",
        dataset={
            "path": "FinanceMTEB/FinTruthQA",
            "revision": "73d569f6d438e430261a924f418ea99981196209",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
