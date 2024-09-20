from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SOPI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SOPI2IRetrieval",
        description="Retrieve product photos of 22634 online products.",
        reference="https://www.florian-schroff.de/publications/CUB-200.pdf",
        dataset={
            "path": "JamieSJS/stanford-online-products",
            "revision": "34e35aeeb47a240e9a60609efa21dd00d6e2478f",
        },
        type="Retrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{oh2016deep,
  title={Deep metric learning via lifted structured feature embedding},
  author={Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4004--4012},
  year={2016}
}
        """,
        descriptive_stats={
            "n_samples": {"default": 120053},
        },
    )
    skip_first_result = True
