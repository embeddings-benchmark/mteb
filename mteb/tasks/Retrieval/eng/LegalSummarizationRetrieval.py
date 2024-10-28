from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalSummarization(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalSummarization",
        description="The dataset consistes of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.",
        reference="https://github.com/lauramanor/legal_summarization",
        dataset={
            "path": "mteb/legal_summarization",
            "revision": "3bb1a05c66872889662af04c5691c14489cebd72",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation="""@inproceedings{manor-li-2019-plain,
    title = "Plain {E}nglish Summarization of Contracts",
    author = "Manor, Laura  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2019",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-2201",
    pages = "1--11",
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 606.1643835616438,
                    "average_query_length": 103.19014084507042,
                    "num_documents": 438,
                    "num_queries": 284,
                    "average_relevant_docs_per_query": 1.545774647887324,
                }
            },
        },
    )
