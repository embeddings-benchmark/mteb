from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SiswatiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SiswatiNewsClassification",
        description="Siswati News Classification Dataset",
        reference="https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news",
        dataset={
            "path": "isaacchung/siswati-news",
            "revision": "f5502326c4e48adc99b18b1582f68b8fb5e7ec30",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ssw-Latn"],
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
        superseded_by="SiswatiNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"title": "text"})


class SiswatiNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SiswatiNewsClassification.v2",
        description="Siswati News Classification Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news",
        dataset={
            "path": "mteb/siswati_news",
            "revision": "e316774d8bbaa9b43858d093ea0f1eb38c6a9b4c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ssw-Latn"],
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
        adapted_from=["SiswatiNewsClassification"],
    )
