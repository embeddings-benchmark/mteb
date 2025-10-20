from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Code1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Code1Retrieval",
        description="Code retrieval dataset with programming questions paired with C/Python/Go/Ruby code snippets for multi-language code retrieval evaluation.",
        reference="https://huggingface.co/datasets/mteb-private/Code1Retrieval-sample",
        dataset={
            "path": "mteb-private/Code1Retrieval",
            "revision": "94d25599a7e0221484f31749448e5ea217484e41",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="bsd-3-clause",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
