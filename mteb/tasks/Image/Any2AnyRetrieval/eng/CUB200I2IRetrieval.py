from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CUB200I2I(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CUB200I2IRetrieval",
        description="Retrieve bird images from 200 classes.",
        reference="https://www.florian-schroff.de/publications/CUB-200.pdf",
        dataset={
            "path": "isaacchung/cub200_retrieval",
            "revision": "ad08c1307b15a226bf1b64e62656a17f1f85f7ec",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{article,
  author = {Welinder, Peter and Branson, Steve and Mita, Takeshi and Wah, Catherine and Schroff, Florian and Belongie, Serge and Perona, Pietro},
  month = {09},
  pages = {},
  title = {Caltech-UCSD Birds 200},
  year = {2010},
}
""",
        descriptive_stats={
            "n_samples": {"default": 5794},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1074.894348894349,
                    "average_query_length": 77.06142506142506,
                    "num_documents": 5794,
                    "num_queries": 5794,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
    skip_first_result = True
