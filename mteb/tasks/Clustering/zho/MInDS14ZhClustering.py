from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class MInDS14ZhClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MInDS14ZhClustering",
        description="MINDS-14 is a dataset for intent detection in e-banking, covering 14 intents across 14 languages.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "FinanceMTEB/MInDS-14-zh",
            "revision": "f42bc3bba1506f41174f2457fc08ec82ab0de162",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
    )
