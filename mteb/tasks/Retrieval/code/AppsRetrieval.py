from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class AppsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AppsRetrieval",
        description="The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://arxiv.org/abs/2105.09938",
        dataset={
            "path": "CoIR-Retrieval/apps",
            "revision": "f22508f96b7a36c2415181ed8bb76f76e04ae2d5",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-05-20", "2021-05-20"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{hendrycksapps2021,
          title={Measuring Coding Challenge Competence With APPS},
          author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
          journal={NeurIPS},
          year={2021}
        }""",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000,
            },
            "test": {
                "average_document_length": 575.0086708499715,
                "average_query_length": 1669.8284196547145,
                "num_documents": 8765,
                "num_queries": 3765,
                "average_relevant_docs_per_query": 1.0,
            },
        },
    )
