from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class METI2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="METI2IRetrieval",
        description="Retrieve photos of more than 224k artworks.",
        reference="https://arxiv.org/abs/2202.01747",
        dataset={
            "path": "JamieSJS/met",
            "revision": "0603c2ff57f406191e1510d15d3de82db2f720c9",
        },
        type="Retrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2021-12-06", "2021-12-14"),  # conference dates
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{ypsilantis2021met,
  title={The met dataset: Instance-level recognition for artworks},
  author={Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
        """,
        descriptive_stats={
            # "n_samples": {"default": 397121},
        },
    )
    skip_first_result = True
