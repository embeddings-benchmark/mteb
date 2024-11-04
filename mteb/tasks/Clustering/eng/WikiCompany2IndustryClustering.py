from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikiCompany2IndustryClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiCompany2IndustryClustering",
        description="Clustering the related industry domain according to the company description.",
        reference="https://aclanthology.org/W18-6532.pdf",
        dataset={
            "path": "FinanceMTEB/WikiCompany2Industry-en",
            "revision": "9b7c45122b764fc1e09c5b29cee887cfa4f8f395",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
    )
