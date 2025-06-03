from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoyageMMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="VoyageMMarcoReranking",
        description="a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite",
        reference="https://arxiv.org/abs/2312.16144",
        dataset={
            "path": "bclavie/mmarco-japanese-hard-negatives",
            "revision": "e25c91bc31859606507a968559ab1de0f472d007",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="map",
        date=("2016-12-01", "2023-12-23"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        prompt="Given a Japanese search query, retrieve web passages that answer the question",
        bibtex_citation=r"""
@misc{clavié2023jacolbert,
  archiveprefix = {arXiv},
  author = {Benjamin Clavié},
  eprint = {2312.16144},
  title = {JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report},
  year = {2023},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column(
            "positives", "positive"
        ).rename_column("negatives", "negative")
        # 391,061 dataset size
        self.dataset["test"] = self.dataset.pop("train").train_test_split(
            test_size=2048, seed=self.seed
        )["test"]
