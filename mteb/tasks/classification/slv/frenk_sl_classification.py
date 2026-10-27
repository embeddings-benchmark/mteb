from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrenkSlClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkSlClassification",
        description="Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.",
        dataset={
            "path": "mteb/FrenkSlClassification",
            "revision": "65ee787b7cc409d51ec6ee4d003596b66cdb70f3",
        },
        reference="https://doi.org/10.1007/978-3-030-27947-9_9",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slv-Latn"],
        main_score="accuracy",
        date=("2021-05-28", "2021-05-28"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{ljubešić2019frenk,
  author = {Ljubešić, Nikola and Fišer, Darja and Erjavec, Tomaž},
  booktitle = {Text, Speech, and Dialogue},
  doi = {10.1007/978-3-030-27947-9_9},
  isbn = {9783030279479},
  issn = {1611-3349},
  pages = {103–114},
  publisher = {Springer International Publishing},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://doi.org/10.1007/978-3-030-27947-9_9},
  year = {2019},
}
""",
        superseded_by="FrenkSlClassification.v2",
    )


class FrenkSlClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkSlClassification.v2",
        description="Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/frenk_sl",
            "revision": "3b69facc14651fbd152fda173683a7ecf9125b82",
        },
        reference="https://doi.org/10.1007/978-3-030-27947-9_9",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slv-Latn"],
        main_score="accuracy",
        date=("2021-05-28", "2021-05-28"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{ljubešić2019frenk,
  author = {Ljubešić, Nikola and Fišer, Darja and Erjavec, Tomaž},
  booktitle = {Text, Speech, and Dialogue},
  doi = {10.1007/978-3-030-27947-9_9},
  isbn = {9783030279479},
  issn = {1611-3349},
  pages = {103–114},
  publisher = {Springer International Publishing},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://doi.org/10.1007/978-3-030-27947-9_9},
  year = {2019},
}
""",
        adapted_from=["FrenkSlClassification"],
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
