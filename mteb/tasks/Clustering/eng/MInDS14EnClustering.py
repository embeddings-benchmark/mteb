from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class MInDS14EnClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MInDS14EnClustering",
        description="MINDS-14 is a dataset for intent detection in e-banking, covering 14 intents across 14 languages.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "FinanceMTEB/MInDS-14-en",
            "revision": "141ac6a9010b851452a9327edfda190d37399b15",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
    )
