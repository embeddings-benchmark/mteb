from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrieval",
        dataset={
            "path": "mteb/quora",
            "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            " question, find other (duplicate) questions."
        ),
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{quora-question-pairs,
    author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
    title = {Quora Question Pairs},
    publisher = {Kaggle},
    year = {2017},
    url = {https://kaggle.com/competitions/quora-question-pairs}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 62.158154708747425,
                    "average_query_length": 51.5342,
                    "num_documents": 522931,
                    "num_queries": 5000,
                    "average_relevant_docs_per_query": 1.5252,
                },
                "test": {
                    "average_document_length": 62.158154708747425,
                    "average_query_length": 51.5396,
                    "num_documents": 522931,
                    "num_queries": 10000,
                    "average_relevant_docs_per_query": 1.5675,
                },
            },
        },
    )
