from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class ComplaintsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ComplaintsClustering",
        description="The Consumer Complaint Database is a collection of complaints about consumer financial products and services that sent to companies for response..",
        reference="https://huggingface.co/datasets/CFPB/consumer-finance-complaints",
        dataset={
            "path": "FinanceMTEB/Complaints",
            "revision": "6704122294b7693f5e544cdde1e4a3e80b291b76",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
    )
