from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["swa-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2023-05-01"),
        dialect=[],
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{davis2020swahili,
  author = {Davis, David},
  doi = {10.5281/zenodo.5514203},
  publisher = {Zenodo},
  title = {Swahili: News Classification Dataset (0.2)},
  url = {https://doi.org/10.5281/zenodo.5514203},
  year = {2020},
}
""",
        superseded_by="SwahiliNewsClassification.v2",
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"content": "text", "category": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class SwahiliNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwahiliNewsClassification.v2",
        description="Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/Mollel/SwahiliNewsClassification",
        dataset={
            "path": "mteb/swahili_news",
            "revision": "d929055f41849d5bc3533c07d978fcfbc89d6a4e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swa-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2023-05-01"),
        dialect=[],
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{davis2020swahili,
  author = {Davis, David},
  doi = {10.5281/zenodo.5514203},
  publisher = {Zenodo},
  title = {Swahili: News Classification Dataset (0.2)},
  url = {https://doi.org/10.5281/zenodo.5514203},
  year = {2020},
}
""",
        adapted_from=["SwahiliNewsClassification"],
    )

    def dataset_transform(self) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
