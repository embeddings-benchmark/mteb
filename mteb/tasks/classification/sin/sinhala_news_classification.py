from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SinhalaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SinhalaNewsClassification",
        description="This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).",
        dataset={
            "path": "mteb/SinhalaNewsClassification",
            "revision": "a86de19f6b79cf464f2bb5ca71d89cdbf41eeb9f",
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
@article{de2015sinhala,
  author = {de Silva, Nisansa},
  journal = {ResearchGate},
  title = {Sinhala text classification: observations from the perspective of a resource poor language},
  year = {2015},
}

@inproceedings{dhananjaya2022bertifying,
  author = {Dhananjaya, Vinura and Demotte, Piyumal and Ranathunga, Surangika and Jayasena, Sanath},
  booktitle = {Proceedings of the thirteenth language resources and evaluation conference},
  pages = {7377--7385},
  title = {BERTifying Sinhala-a comprehensive analysis of pre-trained language models for Sinhala text classification},
  year = {2022},
}
""",
        superseded_by="SinhalaNewsClassification.v2",
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
@article{de2015sinhala,
  author = {de Silva, Nisansa},
  journal = {ResearchGate},
  title = {Sinhala text classification: observations from the perspective of a resource poor language},
  year = {2015},
}

@inproceedings{dhananjaya2022bertifying,
  author = {Dhananjaya, Vinura and Demotte, Piyumal and Ranathunga, Surangika and Jayasena, Sanath},
  booktitle = {Proceedings of the thirteenth language resources and evaluation conference},
  pages = {7377--7385},
  title = {BERTifying Sinhala-a comprehensive analysis of pre-trained language models for Sinhala text classification},
  year = {2022},
}
""",
        adapted_from=["SinhalaNewsClassification"],
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
