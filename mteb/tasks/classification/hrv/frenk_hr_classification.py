from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrenkHrClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkHrClassification",
        description="Croatian subset of the FRENK dataset",
        dataset={
            "path": "mteb/FrenkHrClassification",
            "revision": "eb5117a28f51d6a4e613590174c39b530e275e53",
        },
        reference="https://doi.org/10.1007/978-3-030-27947-9_9",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hrv-Latn"],
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
        superseded_by="FrenkHrClassification.v2",
    )


class FrenkHrClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkHrClassification.v2",
        description="Croatian subset of the FRENK dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/frenk_hr",
            "revision": "09f90d0bee34d5e703caed26737166591a8f12b9",
        },
        reference="https://doi.org/10.1007/978-3-030-27947-9_9",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hrv-Latn"],
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
        adapted_from=["FrenkHrClassification"],
    )
