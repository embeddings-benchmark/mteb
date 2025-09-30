from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class EnglishFinance2Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishFinance2Retrieval",
        description="Financial performance retrieval dataset with queries about stock performance, S&P 500 comparisons, and railroad industry metrics.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishFinance2Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishFinance2Retrieval",
            "revision": "346d5039b9ec75a7b80f8ff008d5ca3df126f5aa",
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
