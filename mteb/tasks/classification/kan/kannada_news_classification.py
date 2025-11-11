from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KannadaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KannadaNewsClassification",
        description="The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.",
        dataset={
            "path": "Akash190104/kannada_news_classification",
            "revision": "a470711069906ac0a559defec3b89cb3725601bd",
        },
        reference="https://github.com/goru001/nlp-for-kannada",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kan-Knda"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
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
        superseded_by="KannadaNewsClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headline", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class KannadaNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KannadaNewsClassification.v2",
        description="The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/kannada_news",
            "revision": "bf9c88b5bd4e5b349a39492e5298a928ab509a92",
        },
        reference="https://github.com/goru001/nlp-for-kannada",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kan-Knda"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
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
        adapted_from=["KannadaNewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
