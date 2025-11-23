from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class UrduRomanSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UrduRomanSentimentClassification",
        description="The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)",
        reference="https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set",
        dataset={
            "path": "mteb/UrduRomanSentimentClassification",
            "revision": "905c1121c002c4b9adc4ebc5faaf4d6f50d1b1ee",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2018-01-01", "2018-08-28"),
        eval_splits=["train"],
        eval_langs=["urd-Latn"],
        main_score="f1",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{misc_roman_urdu_data_set_458,
  author = {Sharf,Zareen},
  howpublished = {UCI Machine Learning Repository},
  note = {{DOI}: https://doi.org/10.24432/C58325},
  title = {{Roman Urdu Data Set}},
  year = {2018},
}
""",
        superseded_by="UrduRomanSentimentClassification.v2",
    )


class UrduRomanSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UrduRomanSentimentClassification.v2",
        description="The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral) This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set",
        dataset={
            "path": "mteb/urdu_roman_sentiment",
            "revision": "fe3ea6b93097e7a2eb1356ad3665fd01667ac6be",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2018-01-01", "2018-08-28"),
        eval_splits=["test"],
        eval_langs=["urd-Latn"],
        main_score="f1",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{misc_roman_urdu_data_set_458,
  author = {Sharf,Zareen},
  howpublished = {UCI Machine Learning Repository},
  note = {{DOI}: https://doi.org/10.24432/C58325},
  title = {{Roman Urdu Data Set}},
  year = {2018},
}
""",
        adapted_from=["UrduRomanSentimentClassification"],
    )
