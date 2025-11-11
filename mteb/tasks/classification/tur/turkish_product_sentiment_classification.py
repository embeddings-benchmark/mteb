from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishProductSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishProductSentimentClassification",
        description="Turkish Product Review Dataset",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "asparius/Turkish-Product-Review",
            "revision": "ad861e463abda351ff65ca5ac0cc5985afe9eb99",
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
        superseded_by="TurkishProductSentimentClassification.v2",
    )


class TurkishProductSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishProductSentimentClassification.v2",
        description="Turkish Product Review Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf",
        dataset={
            "path": "mteb/turkish_product_sentiment",
            "revision": "c846c08821e2ca649929a5562953c0466cd44736",
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
        adapted_from=["TurkishProductSentimentClassification"],
    )
