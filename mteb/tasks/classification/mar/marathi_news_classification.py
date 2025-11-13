from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MarathiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MarathiNewsClassification",
        description="A Marathi dataset for 3-class classification of Marathi news articles",
        reference="https://github.com/goru001/nlp-for-marathi",
        dataset={
            "path": "mlexplorer008/marathi_news_classification",
            "revision": "7640cf8132cca1f99995ac71512a670e3c965cf1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mar-Deva"],
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
        superseded_by="MarathiNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headline": "text"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)


class MarathiNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MarathiNewsClassification.v2",
        description="A Marathi dataset for 3-class classification of Marathi news articles This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/goru001/nlp-for-marathi",
        dataset={
            "path": "mteb/marathi_news",
            "revision": "97932a2f3b75d7bd9fae0d212975c1a1568935eb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mar-Deva"],
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
        adapted_from=["MarathiNewsClassification"],
    )
