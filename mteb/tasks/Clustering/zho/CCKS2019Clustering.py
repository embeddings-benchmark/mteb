from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class CCKS2019Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="CCKS2019Clustering",
        description="Clustering of financial events.",
        reference="https://arxiv.org/abs/2003.03875",
        dataset={
            "path": "FinanceMTEB/CCKS2019",
            "revision": "3ee6454e35b145c3c17413f0e3337b39fffdb1d5",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
    )
