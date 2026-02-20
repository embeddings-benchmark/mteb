from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EnglishFinance3Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishFinance3Retrieval",
        description="Personal finance Q&A retrieval dataset with questions about tax codes, business expenses, and financial advice.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishFinance3Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishFinance3Retrieval",
            "revision": "5ed0bc0fffa309e99400ea800397931749d12cd2",
        },
        type="Retrieval",
        category="t2t",
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
