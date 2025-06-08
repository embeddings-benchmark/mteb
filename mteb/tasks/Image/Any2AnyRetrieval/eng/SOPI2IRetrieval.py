from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SOPI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SOPI2IRetrieval",
        description="Retrieve product photos of 22634 online products.",
        reference="https://paperswithcode.com/dataset/stanford-online-products",
        dataset={
            "path": "JamieSJS/stanford-online-products",
            "revision": "0b3a1622902e6258425e673405bdfb1e5dfa8618",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2019-07-17", "2019-07-17"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{oh2016deep,
  author = {Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {4004--4012},
  title = {Deep metric learning via lifted structured feature embedding},
  year = {2016},
}
""",
        descriptive_stats={
            "n_samples": {"test": 120053},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 120053,
                    "num_queries": 120053,
                    "average_relevant_docs_per_query": 7,
                }
            },
        },
    )
    skip_first_result = True
