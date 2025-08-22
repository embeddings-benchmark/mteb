from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MSCOCOI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MSCOCOI2TRetrieval",
        description="Retrieve captions based on images.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48",
        dataset={
            "path": "MRBench/mbeir_mscoco_task3",
            "revision": "cca3a3e223763e6519a4d68936bc9279034d75d2",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{lin2014microsoft,
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle = {Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  organization = {Springer},
  pages = {740--755},
  title = {Microsoft coco: Common objects in context},
  year = {2014},
}
""",
        prompt={
            "query": "Find an image caption describing the following everyday image."
        },
        descriptive_stats={
            "n_samples": {"test": 5000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 30.94235294117647,
                    "average_query_length": 131.56569965870307,
                    "num_documents": 24809,
                    "num_queries": 5000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
