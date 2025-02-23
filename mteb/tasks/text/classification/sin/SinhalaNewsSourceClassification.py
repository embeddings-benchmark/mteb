from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_classification import AbsTextClassification


class SinhalaNewsSourceClassification(AbsTextClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsSourceClassification",
        description="This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).",
        dataset={
            "path": "NLPC-UOM/Sinhala-News-Source-classification",
            "revision": "ac4d14eeb68efbef95e247542d4432ce674faeb1",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2021-02-17", "2022-08-20"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @article{dhananjaya2022,
            author    = {Dhananjaya et al.},
            title     = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
            journal   = {Year of Publication},
            year      = {2022},
            }""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("comment", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
