from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JapaneseCode1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JapaneseCode1Retrieval",
        description="Japanese code retrieval dataset. Japanese natural language queries paired with Python code snippets for cross-lingual code retrieval evaluation.",
        reference="https://huggingface.co/datasets/mteb-private/JapaneseCode1Retrieval-sample",
        dataset={
            "path": "mteb-private/JapaneseCode1Retrieval",
            "revision": "fc4cb6390055e65490dfc42526e1d6a379e8cd86",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
