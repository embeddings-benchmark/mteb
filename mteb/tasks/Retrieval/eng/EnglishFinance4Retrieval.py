from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class EnglishFinance4Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishFinance4Retrieval",
        description="Personal finance advice retrieval dataset with questions about car financing, investment strategies, and financial planning.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishFinance4Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishFinance4Retrieval",
            "revision": "2fdb7001309f897d50d38d196a3fd0f03c913810",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Written", "Non-fiction"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
