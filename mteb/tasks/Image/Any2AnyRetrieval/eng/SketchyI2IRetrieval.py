from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SketchyI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SketchyI2IRetrieval",
        description="Retrieve photos from sketches.",
        reference="https://arxiv.org/abs/2202.01747",
        dataset={
            "path": "JamieSJS/sketchy",
            "revision": "c8b8c1b7a2f0a92f1bfaaa1c9afc22aa42c61d5b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2021-12-06", "2021-12-14"),  # conference dates
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{ypsilantis2021met,
  author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {The met dataset: Instance-level recognition for artworks},
  year = {2021},
}
""",
        descriptive_stats={
            "n_samples": {"test": 452886},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 7.24,
                    "num_documents": 25000,
                    "num_queries": 452886,
                    "average_relevant_docs_per_query": 3623.0,
                }
            },
        },
    )
    skip_first_result = False
