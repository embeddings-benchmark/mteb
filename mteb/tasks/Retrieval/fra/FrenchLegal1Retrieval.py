from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenchLegal1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FrenchLegal1Retrieval",
        description="French legal document retrieval dataset with queries about administrative law, court decisions, and legal proceedings.",
        reference="https://huggingface.co/datasets/mteb-private/FrenchLegal1Retrieval-sample",
        dataset={
            "path": "mteb-private/FrenchLegal1Retrieval",
            "revision": "44c4be8bfc12406025bf12295cd260c5ed189848",  # Updated with latest sample commit 44c4be8b
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
