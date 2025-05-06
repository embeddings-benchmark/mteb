from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class OKVQAIT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="OKVQAIT2TRetrieval",
        description="Retrieval a Wiki passage to answer query about an image.",
        reference="https://okvqa.allenai.org",
        dataset={
            "path": "izhx/UMRB-OKVQA",
            "revision": "96a84a043f5465893670cf616f90e64086c0417a",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_10",
        date=("2019-01-01", "2020-07-29"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{marino2019ok,
  author = {Marino, Kenneth and Rastegari, Mohammad and Farhadi, Ali and Mottaghi, Roozbeh},
  booktitle = {Proceedings of the IEEE/cvf conference on computer vision and pattern recognition},
  pages = {3195--3204},
  title = {Ok-vqa: A visual question answering benchmark requiring external knowledge},
  year = {2019},
}
""",
        prompt={
            "query": "Retrieve documents that provide an answer to the question alongside the image."
        },
        descriptive_stats={
            "n_samples": {"test": 5046},
            "avg_character_length": {
                "test": {
                    "average_document_length": 41.7072929052715,
                    "average_query_length": 631.7119703796849,
                    "num_documents": 114516,
                    "num_queries": 5046,
                    "average_relevant_docs_per_query": 7.426674593737614,
                }
            },
        },
    )
