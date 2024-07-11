from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RuBQRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuBQRetrieval",
        dataset={
            "path": "ai-forever/rubq-retrieval",
            "revision": "e19b6ffa60b3bc248e0b41f4cc37c26a55c2a67b",
        },
        description="Paragraph retrieval based on RuBQ 2.0. Retrieve paragraphs from Wikipedia that answer the question.",
        reference="https://openreview.net/pdf?id=P5UQFFoQ4PJ",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2001-01-01", "2021-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{RuBQ2021,
        title={RuBQ 2.0: An Innovated Russian Question Answering Dataset},
        author={Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
        booktitle={ESWC},
        year={2021},
        pages={532--547}
        }""",
        stats={
            "n_samples": {"test": 2845},
            "avg_character_length": {
                "test": {
                    "average_document_length": 448.94659134903037,
                    "average_query_length": 45.29609929078014,
                    "num_documents": 56826,
                    "num_queries": 1692,
                    "average_relevant_docs_per_query": 1.6814420803782506,
                }
            },
        },
    )
