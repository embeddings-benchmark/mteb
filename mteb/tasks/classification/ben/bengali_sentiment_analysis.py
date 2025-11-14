from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BengaliSentimentAnalysis(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliSentimentAnalysis",
        description="dataset contains 3307 Negative reviews and 8500 Positive reviews collected and manually annotated from Youtube Bengali drama.",
        reference="https://data.mendeley.com/datasets/p6zc7krs37/4",
        dataset={
            "path": "Akash190104/bengali_sentiment_analysis",
            "revision": "a4b3685b1854cc26c554dda4c7cb918a36a6fb6c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ben-Beng"],
        main_score="f1",
        date=("2020-06-24", "2020-11-26"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sazzed2020cross,
  author = {Sazzed, Salim},
  booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
  pages = {50--60},
  title = {Cross-lingual sentiment classification in low-resource Bengali language},
  year = {2020},
}
""",
        superseded_by="BengaliSentimentAnalysis.v2",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )


class BengaliSentimentAnalysisV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliSentimentAnalysis.v2",
        description="dataset contains 2854 Negative reviews and 7238 Positive reviews collected and manually annotated from Youtube Bengali drama. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)",
        reference="https://data.mendeley.com/datasets/p6zc7krs37/4",
        dataset={
            "path": "mteb/bengali_sentiment_analysis",
            "revision": "23edb78a3dd297a4d92f9c011a0503be0c0949d0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ben-Beng"],
        main_score="f1",
        date=("2020-06-24", "2020-11-26"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sazzed2020cross,
  author = {Sazzed, Salim},
  booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
  pages = {50--60},
  title = {Cross-lingual sentiment classification in low-resource Bengali language},
  year = {2020},
}
""",
    )
