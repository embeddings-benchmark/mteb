from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

# from ....abstasks import AbsTaskClassification
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification

class SwahiliNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwahiliNewsClassification",
        description="Dataset for Swahili News Classification, categorized with 5 domains. Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets",
        reference="https://www.sartify.com/",
        dataset={
            "path": "Mollel/SwahiliNewsClassification",
            "revision": "fadfd9f2989740dc6c9aa020cac97b318ec41cc3",
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
        """,
        n_samples={"train": 2048},
        avg_character_length={"train": 2107.1962725684334},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"content": "text", "category": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
