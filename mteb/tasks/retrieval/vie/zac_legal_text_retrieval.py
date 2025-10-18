from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ZacLegalTextRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ZacLegalTextRetrieval",
        description="Zalo Legal Text documents",
        reference="https://challenge.zalo.ai",
        dataset={
            "path": "GreenNode/zalo-ai-legal-text-retrieval-vn",
            "revision": "910766554633e8da014e88f54988705dde7ecaac",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-03-16", "2025-03-16"),
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # TODO: Add bibtex citation when the paper is published
    )
