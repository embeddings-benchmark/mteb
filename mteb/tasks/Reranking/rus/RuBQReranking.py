from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class RuBQReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="RuBQReranking",
        dataset={
            "path": "ai-forever/rubq-reranking",
            "revision": "2e96b8f098fa4b0950fc58eacadeb31c0d0c7fa2",
        },
        description="Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.",
        reference="https://openreview.net/pdf?id=P5UQFFoQ4PJ",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="map_at_1000",
        date=("2001-01-01", "2021-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
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
        descriptive_stats={
            "n_samples": {"test": 1551},
            "test": {
                "average_document_length": 457.17801158971344,
                "average_query_length": 42.818826563507415,
                "num_documents": 37447,
                "num_queries": 1551,
                "average_relevant_docs_per_query": 1.6776273372018053,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 24.143778207607994,
            },
        },
    )
