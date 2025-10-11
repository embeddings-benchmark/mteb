from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GermanLegal1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanLegal1Retrieval",
        description="German educational regulation retrieval dataset with queries about university capacity calculations and academic administration.",
        reference="https://huggingface.co/datasets/mteb-private/GermanLegal1Retrieval-sample",
        dataset={
            "path": "mteb-private/GermanLegal1Retrieval",
            "revision": "65ea369daff680b77f90b560e7e97d2ab4ec5072",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
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
