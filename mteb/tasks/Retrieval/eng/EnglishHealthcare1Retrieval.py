from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EnglishHealthcare1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnglishHealthcare1Retrieval",
        description="Medical research retrieval dataset with queries about HIV transmission, genetic variants, and biomedical research findings.",
        reference="https://huggingface.co/datasets/mteb-private/EnglishHealthcare1Retrieval-sample",
        dataset={
            "path": "mteb-private/EnglishHealthcare1Retrieval",
            "revision": "393c24e85114d44c43259fb2d1c5639c5d09809d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
