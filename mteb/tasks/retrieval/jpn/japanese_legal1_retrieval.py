from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JapaneseLegal1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JapaneseLegal1Retrieval",
        description="Japanese legal regulation retrieval dataset with queries about government regulations, ministry ordinances, and administrative law.",
        reference="https://huggingface.co/datasets/mteb-private/JapaneseLegal1Retrieval-sample",
        dataset={
            "path": "mteb-private/JapaneseLegal1Retrieval",
            "revision": "d653557fe66bb6af2b0e2adfc371a24554cf11ce",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
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
