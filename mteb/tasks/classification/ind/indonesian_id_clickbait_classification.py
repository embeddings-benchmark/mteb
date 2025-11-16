from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class IndonesianIdClickbaitClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndonesianIdClickbaitClassification",
        dataset={
            "path": "manandey/id_clickbait",
            "revision": "9fa4d0824015fe537ae2c8166781f5c79873da2c",
        },
        description="The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.",
        reference="http://www.sciencedirect.com/science/article/pii/S2352340920311252",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ind-Latn"],
        main_score="f1",
        date=("2020-10-01", "2020-10-01"),
        domains=["News", "Written"],
        dialect=[],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@article{WILLIAM2020106231,
  author = {Andika William and Yunita Sari},
  doi = {https://doi.org/10.1016/j.dib.2020.106231},
  issn = {2352-3409},
  journal = {Data in Brief},
  keywords = {Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification},
  pages = {106231},
  title = {CLICK-ID: A novel dataset for Indonesian clickbait headlines},
  url = {http://www.sciencedirect.com/science/article/pii/S2352340920311252},
  volume = {32},
  year = {2020},
}
""",
        superseded_by="IndonesianIdClickbaitClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.remove_columns(["label"]).rename_columns(
            {"title": "text", "label_score": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class IndonesianIdClickbaitClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndonesianIdClickbaitClassification.v2",
        dataset={
            "path": "mteb/indonesian_id_clickbait",
            "revision": "a54158a1b437a85e1982a70d0c57a69c69d0a5b8",
        },
        description="The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="http://www.sciencedirect.com/science/article/pii/S2352340920311252",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ind-Latn"],
        main_score="f1",
        date=("2020-10-01", "2020-10-01"),
        domains=["News", "Written"],
        dialect=[],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@article{WILLIAM2020106231,
  author = {Andika William and Yunita Sari},
  doi = {https://doi.org/10.1016/j.dib.2020.106231},
  issn = {2352-3409},
  journal = {Data in Brief},
  keywords = {Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification},
  pages = {106231},
  title = {CLICK-ID: A novel dataset for Indonesian clickbait headlines},
  url = {http://www.sciencedirect.com/science/article/pii/S2352340920311252},
  volume = {32},
  year = {2020},
}
""",
        adapted_from=["IndonesianIdClickbaitClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
