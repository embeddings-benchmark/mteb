from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{deSilva2015,
  author = {Nisansa de Silva},
  journal = {Year of Publication},
  title = {Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language},
  year = {2015},
}

@article{dhananjaya2022,
  author = {Dhananjaya et al.},
  journal = {Year of Publication},
  title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
  year = {2022},
}
""",
        superseded_by="SinhalaNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"comments": "text", "labels": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class SinhalaNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsClassification.v2",
        description="This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015). This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/sinhala_news",
            "revision": "e0b6e93ed5f086fe358595dff1aaad9eb877667a",
        },
        reference="https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{deSilva2015,
  author = {Nisansa de Silva},
  journal = {Year of Publication},
  title = {Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language},
  year = {2015},
}

@article{dhananjaya2022,
  author = {Dhananjaya et al.},
  journal = {Year of Publication},
  title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
  year = {2022},
}
""",
        adapted_from=["SinhalaNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
