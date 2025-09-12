from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class EnglishFinance1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishFinance1Retrieval",
        description="Financial document retrieval dataset with queries about stock compensation, corporate governance, and SEC filing content.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishFinance1Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishFinance1Retrieval",
            "revision": "f27cad5a2ba8900b89a85ca26d95de3361a6f8a0",  # Updated with latest sample commit f27cad5a
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Written", "Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
