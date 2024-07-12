from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BengaliDocumentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliDocumentClassification",
        description="Dataset for News Classification, categorized with 13 domains.",
        reference="https://aclanthology.org/2023.eacl-main.4",
        dataset={
            "path": "dialect-ai/shironaam",
            "revision": "1c6e67433da618073295b7c90f1c55fa8e78f35c",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ben-Beng"],
        main_score="accuracy",
        date=("2022-05-01", "2023-05-01"),
        dialect=[],
        domains=["News", "Written"],
        task_subtypes=[],
        license="CC BY-NC-SA 4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{akash-etal-2023-shironaam,
            title = "Shironaam: {B}engali News Headline Generation using Auxiliary Information",
            author = "Akash, Abu Ubaida  and
            Nayeem, Mir Tafseer  and
            Shohan, Faisal Tareque  and
            Islam, Tanvir",
            booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
            month = may,
            year = "2023",
            address = "Dubrovnik, Croatia",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.eacl-main.4",
            pages = "52--67"
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {"test": 1658.1},
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "text", "category": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
