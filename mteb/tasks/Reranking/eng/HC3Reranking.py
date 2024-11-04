from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class HC3Reranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="HC3Reranking",
        description="A human-ChatGPT comparison finance corpus",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/HPC3-reranking",
            "revision": "538a43bb89e86fd40a8d1b9c7e630f8c3aa67a6c",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
    )
