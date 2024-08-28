from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class SyntheticText2SQLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntheticText2SQL",
        description="The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
        dataset={
            "path": "CoIR-Retrieval/synthetic-text2sql",
            "revision": "686b87296c3a0191b5d9415a00526c62db9fce09",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "sql-Code"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@software{gretel-synthetic-text-to-sql-2024,
          author = {Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew},
          title = {{Synthetic-Text-To-SQL}: A synthetic dataset for training language models to generate SQL queries from natural language prompts},
          month = {April},
          year = {2024},
          url = {https://huggingface.co/datasets/gretelai/synthetic-text-to-sql}
        }""",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000,
            },
            "avg_character_length": {
                "test": {
                    "average_document_length": 127.07126054548375,
                    "average_query_length": 82.90582806357888,
                    "num_documents": 105851,
                    "num_queries": 5851,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
