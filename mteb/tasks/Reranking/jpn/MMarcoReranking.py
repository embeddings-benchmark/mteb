from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoyageMMarcoReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VoyageMMarcoReranking",
        description="a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite",
        reference="https://arxiv.org/abs/2312.16144",
        dataset={
            "path": "mteb/VoyageMMarcoReranking",
            "revision": "bd2050c52b480e48c51372b4ec98a1cbbc4515f2",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="map_at_1000",
        date=("2016-12-01", "2023-12-23"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        prompt="Given a Japanese search query, retrieve web passages that answer the question",
        bibtex_citation="""@misc{clavié2023jacolbert,
      title={JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report}, 
      author={Benjamin Clavié},
      year={2023},
      eprint={2312.16144},
      archivePrefix={arXiv},}""",
    )
