from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class FiQA2018Reranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="FiQA2018Reranking",
        description="Financial opinion mining and question answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "FinanceMTEB/FiQA-reranking",
            "revision": "f6934c7980c19a3acb8aeba3b66b4766fbb4b9db",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
    )
