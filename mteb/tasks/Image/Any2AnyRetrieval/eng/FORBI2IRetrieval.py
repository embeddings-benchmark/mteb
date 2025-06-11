from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FORBI2I(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="FORBI2IRetrieval",
        description="Retrieve flat object images from 8 classes.",
        reference="https://github.com/pxiangwu/FORB",
        dataset={
            "path": "isaacchung/forb_retrieval",
            "revision": "26ab4bd972854becada339afc80f5f3ffc047e2b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2022-01-01", "2023-01-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{wu2023forbflatobjectretrieval,
  archiveprefix = {arXiv},
  author = {Pengxiang Wu and Siman Wang and Kevin Dela Rosa and Derek Hao Hu},
  eprint = {2309.16249},
  primaryclass = {cs.CV},
  title = {FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding},
  url = {https://arxiv.org/abs/2309.16249},
  year = {2023},
}
""",
        descriptive_stats={
            "n_samples": {"default": 13250},
            "avg_character_length": {
                "test": {
                    "num_documents": 53984,
                    "num_queries": 13250,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
