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
        reference="https://arxiv.org/abs/1906.02045",
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
@misc{ljubešić2019frenk,
  archiveprefix = {arXiv},
  author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
  eprint = {1906.02045},
  primaryclass = {cs.CL},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://arxiv.org/abs/1906.02045},
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
        reference="https://arxiv.org/abs/1906.02045",
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
@misc{ljubešić2019frenk,
  archiveprefix = {arXiv},
  author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
  eprint = {1906.02045},
  primaryclass = {cs.CL},
  title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
  url = {https://arxiv.org/abs/1906.02045},
  year = {2019},
}
""",
        adapted_from=["FrenkHrClassification"],
    )
