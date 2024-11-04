from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class CCKS2020Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="CCKS2020Clustering",
        description="Clustering of financial events.",
        reference="https://www.biendata.xyz/competition/ccks_2020_4_2",
        dataset={
            "path": "FinanceMTEB/CCKS2020",
            "revision": "747e32650174620b2ab1b21684a7a5b135a0980b",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
    )
