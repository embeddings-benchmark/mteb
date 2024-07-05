from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SwahiliNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwahiliNewsClassification",
        description="Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets",
        reference="https://huggingface.co/datasets/Mollel/SwahiliNewsClassification",
        dataset={
            "path": "Mollel/SwahiliNewsClassification",
            "revision": "24fcf066e6b96f9e0d743e8b79184e0c599f73c3",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["swa-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2023-05-01"),
        form=["written"],
        dialect=[],
        domains=["News"],
        task_subtypes=[],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""
        @inproceedings{davis2020swahili,
        title = "Swahili: News Classification Dataset (0.2)",
        author = "Davis, David",
        year = "2020",
        publisher = "Zenodo",
        doi = "10.5281/zenodo.5514203",
        url = "https://doi.org/10.5281/zenodo.5514203"
        }
        """,
        n_samples={"train": 2048},
        avg_character_length={"train": 2438.2308135942326},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"content": "text", "category": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
