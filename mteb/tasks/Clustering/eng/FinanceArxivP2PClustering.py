from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinanceArxivP2PClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FinanceArxivP2PClustering",
        description="Clustering of titles from arxiv (q-fin).",
        reference="https://arxiv.org/abs/2409.18511v1",
        dataset={
            "path": "FinanceMTEB/FinanceArxiv-p2p",
            "revision": "5cc8768f04d39e9a16b84fc236dec93d8b173b64",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
    )
