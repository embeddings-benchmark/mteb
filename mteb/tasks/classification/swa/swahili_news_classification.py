from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SwahiliNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwahiliNewsClassification",
        description="Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets",
        reference="https://huggingface.co/datasets/Mollel/SwahiliNewsClassification",
        dataset={
            "path": "mteb/SwahiliNewsClassification",
            "revision": "e9b91a8bc13eb2797ef476d72feda71581d0a4fe",
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

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
