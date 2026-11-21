from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class GeorgianFAQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GeorgianFAQRetrieval",
        dataset={
            "path": "mteb/GeorgianFAQRetrieval",
            "revision": "edf45abca6fe559f1417a03d5fe44775013f7142",
        },
        description=(
            "Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kat-Geor"],
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        sample_creation="created",
        reference="https://huggingface.co/datasets/jupyterjazz/georgian-faq",
        date=("2024-05-02", "2024-05-03"),
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
    )
