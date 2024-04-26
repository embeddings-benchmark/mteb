from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SinhalaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsClassification",
        description="This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).",
        dataset={
            "path": "NLPC-UOM/Sinhala-News-Category-classification",
            "revision": "7fb2f514ea683c5282dfec0a9672ece8de90ac50",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="mit",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{deSilva2015,
            author    = {Nisansa de Silva},
            title     = {Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language},
            journal   = {Year of Publication},
            year      = {2015},
            }
            @article{dhananjaya2022,
            author    = {Dhananjaya et al.},
            title     = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
            journal   = {Year of Publication},
            year      = {2022},
            }""",
        n_samples={"train": 3327},
        avg_character_length={"train": 148.04},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"comments": "text", "labels": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
