from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NIGHTSI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NIGHTSI2IRetrieval",
        description="Retrieval identical image to the given image.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html",
        dataset={
            "path": "mteb/mbeir_nights_task4",
            "revision": "c798fa2f5173dc9fe0c727fd0abfd2f94cbc2f23",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Image Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{fu2024dreamsim,
  author = {Fu, Stephanie and Tamir, Netanel and Sundaram, Shobhita and Chai, Lucy and Zhang, Richard and Dekel, Tali and Isola, Phillip},
  journal = {Advances in Neural Information Processing Systems},
  title = {DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data},
  volume = {36},
  year = {2024},
}
""",
        prompt={
            "query": "Find a day-to-day image that looks similar to the provided image."
        },
        superseded_by="NIGHTSI2IRetrieval.v2",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # fixes https://github.com/embeddings-benchmark/mteb/issues/4436
        self.dataset["default"]["test"]["corpus"] = self.dataset["default"]["test"][
            "corpus"
        ].remove_columns("text")
        self.dataset["default"]["test"]["queries"] = self.dataset["default"]["test"][
            "queries"
        ].remove_columns("text")


class NIGHTSI2IRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NIGHTSI2IRetrieval.v2",
        description=(
            "Retrieval identical image to the given image. "
            "Version 2 sets the canonical metric to hit_rate_at_5, matching the "
            "M-BEIR/UniIR source metric (hit-style Recall@5) instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to NIGHTSI2IRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html",
        dataset={
            "path": "mteb/mbeir_nights_task4",
            "revision": "c798fa2f5173dc9fe0c727fd0abfd2f94cbc2f23",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Image Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{fu2024dreamsim,
  author = {Fu, Stephanie and Tamir, Netanel and Sundaram, Shobhita and Chai, Lucy and Zhang, Richard and Dekel, Tali and Isola, Phillip},
  journal = {Advances in Neural Information Processing Systems},
  title = {DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data},
  volume = {36},
  year = {2024},
}
""",
        prompt={
            "query": "Find a day-to-day image that looks similar to the provided image."
        },
        adapted_from=["NIGHTSI2IRetrieval"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # fixes https://github.com/embeddings-benchmark/mteb/issues/4436
        self.dataset["default"]["test"]["corpus"] = self.dataset["default"]["test"][
            "corpus"
        ].remove_columns("text")
        self.dataset["default"]["test"]["queries"] = self.dataset["default"]["test"][
            "queries"
        ].remove_columns("text")
