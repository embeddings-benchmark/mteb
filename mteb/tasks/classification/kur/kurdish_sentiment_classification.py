from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KurdishSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KurdishSentimentClassification",
        description="Kurdish Sentiment Dataset",
        reference="https://link.springer.com/article/10.1007/s10579-023-09716-6",
        dataset={
            "path": "asparius/Kurdish-Sentiment",
            "revision": "f334d90a9f68cc3af78cc2a2ece6a3b69408124c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kur-Arab"],
        main_score="accuracy",
        date=("2023-01-01", "2024-01-02"),
        domains=["Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["Sorani"],
        sample_creation="found",
        bibtex_citation=r"""
@article{article,
  author = {Badawi, Soran and Kazemi, Arefeh and Rezaie, Vali},
  doi = {10.1007/s10579-023-09716-6},
  journal = {Language Resources and Evaluation},
  month = {01},
  pages = {1-20},
  title = {KurdiSent: a corpus for kurdish sentiment analysis},
  year = {2024},
}
""",
        superseded_by="KurdishSentimentClassification.v2",
    )


class KurdishSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KurdishSentimentClassification.v2",
        description="Kurdish Sentiment Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://link.springer.com/article/10.1007/s10579-023-09716-6",
        dataset={
            "path": "mteb/kurdish_sentiment",
            "revision": "f6b00b2a1fcbffd83f10a76c85f246ca750c83d2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kur-Arab"],
        main_score="accuracy",
        date=("2023-01-01", "2024-01-02"),
        domains=["Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["Sorani"],
        sample_creation="found",
        bibtex_citation=r"""
@article{article,
  author = {Badawi, Soran and Kazemi, Arefeh and Rezaie, Vali},
  doi = {10.1007/s10579-023-09716-6},
  journal = {Language Resources and Evaluation},
  month = {01},
  pages = {1-20},
  title = {KurdiSent: a corpus for kurdish sentiment analysis},
  year = {2024},
}
""",
        adapted_from=["KurdishSentimentClassification"],
    )
