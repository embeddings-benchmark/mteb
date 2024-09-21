from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class VisualNewsT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VisualNewsT2IRetrieval",
        description="Retrieve news images with captions.",
        reference="https://aclanthology.org/2021.emnlp-main.542/",
        dataset={
            "path": "MRBench/mbeir_visualnews_task0",
            "revision": "94c519d850dba2b0058c2fc9b5da6142a59aa285",
        },
        type="Retrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{liu2021visual,
  title={Visual News: Benchmark and Challenges in News Image Captioning},
  author={Liu, Fuxiao and Wang, Yinghan and Wang, Tianlu and Ordonez, Vicente},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={6761--6771},
  year={2021}
}""",
        descriptive_stats={
            "n_samples": {"test": 19995},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 542246,
                    "num_queries": 19995,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
