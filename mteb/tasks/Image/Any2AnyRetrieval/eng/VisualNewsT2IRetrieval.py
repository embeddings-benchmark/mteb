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
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{liu2021visual,
  author = {Liu, Fuxiao and Wang, Yinghan and Wang, Tianlu and Ordonez, Vicente},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages = {6761--6771},
  title = {Visual News: Benchmark and Challenges in News Image Captioning},
  year = {2021},
}
""",
        prompt={
            "query": "Identify the news-related image in line with the described event."
        },
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
