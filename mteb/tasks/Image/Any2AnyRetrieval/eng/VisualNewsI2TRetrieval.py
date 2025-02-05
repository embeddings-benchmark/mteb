from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class VisualNewsI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VisualNewsI2TRetrieval",
        description="Retrieval entity-rich captions for news images.",
        reference="https://aclanthology.org/2021.emnlp-main.542/",
        dataset={
            "path": "MRBench/mbeir_visualnews_task3",
            "revision": "aaee58895a66e4d619168849267ed2bb40d37043",
        },
        type="Any2AnyRetrieval",
        category="i2t",
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
        bibtex_citation="""@inproceedings{liu2021visual,
  title={Visual News: Benchmark and Challenges in News Image Captioning},
  author={Liu, Fuxiao and Wang, Yinghan and Wang, Tianlu and Ordonez, Vicente},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={6761--6771},
  year={2021}
}""",
        prompt={"query": "Find a caption for the news in the given photo."},
        descriptive_stats={
            "n_samples": {"test": 20000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 537568,
                    "num_queries": 20000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
