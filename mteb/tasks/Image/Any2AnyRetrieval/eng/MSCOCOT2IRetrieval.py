from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MSCOCOT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MSCOCOT2IRetrieval",
        description="Retrieve images based on captions.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48",
        dataset={
            "path": "MRBench/mbeir_mscoco_task0",
            "revision": "cfe15bd2791dde5f8f20aebecf0b4eb3812972d6",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="t2i",
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
        bibtex_citation="""@inproceedings{lin2014microsoft,
        title={Microsoft coco: Common objects in context},
        author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
        booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
        pages={740--755},
        year={2014},
        organization={Springer}
        }""",
        prompt={"query": "Identify the image showcasing the described everyday scene."},
        descriptive_stats={
            "n_samples": {"test": 24809},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 5000,
                    "num_queries": 24809,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
