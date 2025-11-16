from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishMovieSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishMovieSentimentClassification",
        description="Turkish Movie Review Dataset",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "asparius/Turkish-Movie-Review",
            "revision": "409a4415cce5f6bcfca6d5f3ca3c408211ca00b3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        date=("2013-01-01", "2013-08-11"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Demirtas2013CrosslingualPD,
  author = {Erkin Demirtas and Mykola Pechenizkiy},
  booktitle = {wisdom},
  title = {Cross-lingual polarity detection with machine translation},
  url = {https://api.semanticscholar.org/CorpusID:3912960},
  year = {2013},
}
""",
        superseded_by="TurkishMovieSentimentClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class TurkishMovieSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishMovieSentimentClassification.v2",
        description="Turkish Movie Review Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "mteb/turkish_movie_sentiment",
            "revision": "8ef5ce93ff2504de7fc46776317b78bdd8db47f2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        date=("2013-01-01", "2013-08-11"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Demirtas2013CrosslingualPD,
  author = {Erkin Demirtas and Mykola Pechenizkiy},
  booktitle = {wisdom},
  title = {Cross-lingual polarity detection with machine translation},
  url = {https://api.semanticscholar.org/CorpusID:3912960},
  year = {2013},
}
""",
        adapted_from=["TurkishMovieSentimentClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
