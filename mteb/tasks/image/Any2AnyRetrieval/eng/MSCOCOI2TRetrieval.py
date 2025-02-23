from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        task_subtypes=["image Text Retrieval"],
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
        prompt={
            "query": "Find an image caption describing the following everyday image."
        },
    )
