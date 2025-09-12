from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class EnglishFinance3Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishFinance3Retrieval",
        description="Personal finance Q&A retrieval dataset with questions about tax codes, business expenses, and financial advice.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishFinance3Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishFinance3Retrieval",
            "revision": "4052756ac05fcb766ca97abd682c8f2a50e358d6",  # Updated with latest sample commit 4052756a
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
