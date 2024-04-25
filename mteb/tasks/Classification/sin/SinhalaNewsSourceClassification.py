from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SinhalaNewsSourceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsSourceClassification",
        description="This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).",
        dataset={
            "path": "NLPC-UOM/Sinhala-News-Source-classification",
            "revision": "7fb2f514ea683c5282dfec0a9672ece8de90ac50",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2021-02-17", "2022-08-20"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="mit",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @article{dhananjaya2022,
            author    = {Dhananjaya et al.},
            title     = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
            journal   = {Year of Publication},
            year      = {2022},
            }""",
        n_samples={"train": 24094},
        avg_character_length={"train": 56.08},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("comment", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
