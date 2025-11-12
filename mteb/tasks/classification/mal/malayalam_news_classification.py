from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MalayalamNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalayalamNewsClassification",
        description="A Malayalam dataset for 3-class classification of Malayalam news articles",
        reference="https://github.com/goru001/nlp-for-malyalam",
        dataset={
            "path": "mlexplorer008/malayalam_news_classification",
            "revision": "666f63bba2387456d8f846ea4d0565181bd47b81",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mal-Mlym"],
        main_score="accuracy",
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
        superseded_by="MalayalamNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"headings": "text"})


class MalayalamNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MalayalamNewsClassification.v2",
        description="A Malayalam dataset for 3-class classification of Malayalam news articles This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/goru001/nlp-for-malyalam",
        dataset={
            "path": "mteb/malayalam_news",
            "revision": "2bb37780ab4a68cb0b28a902059463563b2dbab9",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["mal-Mlym"],
        main_score="accuracy",
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
        adapted_from=["MalayalamNewsClassification"],
    )
