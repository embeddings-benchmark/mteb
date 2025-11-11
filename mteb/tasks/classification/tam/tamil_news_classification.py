from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TamilNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TamilNewsClassification",
        description="A Tamil dataset for 6-class classification of Tamil news articles",
        reference="https://github.com/vanangamudi/tamil-news-classification",
        dataset={
            "path": "mlexplorer008/tamil_news_classification",
            "revision": "bb34dd6690cf17aa731d75d45388c5801b8c4e4b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tam-Taml"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{kunchukuttan2020indicnlpcorpus,
  author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
  journal = {arXiv preprint arXiv:2005.00085},
  title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
  year = {2020},
}
""",
        superseded_by="TamilNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"NewsInTamil": "text", "Category": "label"}
        )
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)


class TamilNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TamilNewsClassification.v2",
        description="A Tamil dataset for 6-class classification of Tamil news articles This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/vanangamudi/tamil-news-classification",
        dataset={
            "path": "mteb/tamil_news",
            "revision": "b417dba1f5a3143f8325b6b6fb585ab4a57c03a0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tam-Taml"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{kunchukuttan2020indicnlpcorpus,
  author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
  journal = {arXiv preprint arXiv:2005.00085},
  title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
  year = {2020},
}
""",
        adapted_from=["TamilNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
