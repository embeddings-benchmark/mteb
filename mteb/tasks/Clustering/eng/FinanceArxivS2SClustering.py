from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinanceArxivS2SClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FinanceArxivS2SClustering",
        description="Clustering of titles from arxiv (q-fin).",
        # reference=" ",
        dataset={
            "path": "FinanceMTEB/FinanceArxiv-s2s",
            "revision": "78f66d3bbea9b1d7f11df84bc55b21b24302dcee",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
    )
