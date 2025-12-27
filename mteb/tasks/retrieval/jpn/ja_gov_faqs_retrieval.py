from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JaGovFaqsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="JaGovFaqsRetrieval",
        description="JaGovFaqs is a dataset consisting of FAQs manually extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/JaGovFaqsRetrieval",
            "revision": "0726b2af53c907628067871e2e7c84e0d8e099c2",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
