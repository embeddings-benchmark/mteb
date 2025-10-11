from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GermanHealthcare1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanHealthcare1Retrieval",
        description="German medical consultation retrieval dataset with patient questions and doctor responses about various health conditions.",
        reference="https://huggingface.co/datasets/mteb-private/GermanHealthcare1Retrieval-sample",
        dataset={
            "path": "mteb-private/GermanHealthcare1Retrieval",
            "revision": "53e9a6fb88b48b7513e9d2cc2218e3415f4e45f8",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
