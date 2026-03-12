from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class IsiZuluNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IsiZuluNewsClassification",
        description="isiZulu News Classification Dataset",
        reference="https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news",
        dataset={
            "path": "mteb/IsiZuluNewsClassification",
            "revision": "692de64a824f0620161b6fc0b2c20f84d2b07b6f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["zul-Latn"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Madodonga_Marivate_Adendorff_2023,
  author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
  doi = {10.55492/dhasa.v4i01.4449},
  month = {Jan.},
  title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
  url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
  volume = {4},
  year = {2023},
}
""",
        superseded_by="IsiZuluNewsClassification.v2",
    )


class IsiZuluNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IsiZuluNewsClassification.v2",
        description="isiZulu News Classification Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news",
        dataset={
            "path": "mteb/isi_zulu_news",
            "revision": "45708aaaf9c6133227ea5db5cf26571facb9ccdb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["zul-Latn"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Madodonga_Marivate_Adendorff_2023,
  author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
  doi = {10.55492/dhasa.v4i01.4449},
  month = {Jan.},
  title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
  url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
  volume = {4},
  year = {2023},
}
""",
        adapted_from=["IsiZuluNewsClassification"],
    )
