from __future__ import annotations

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class DuEEFinRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuEEFinRetrieval",
        description="Financial news bulletin event quiz dataset.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DuEE-fin",
            "revision": "1129a95b1b81a298497929be04e8e681da48eba4",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
    )
