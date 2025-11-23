from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MovieReviewSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MovieReviewSentimentClassification",
        dataset={
            "path": "tblard/allocine",
            "revision": "a4654f4896408912913a62ace89614879a549287",
        },
        description="The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.",
        reference="https://github.com/TheophileBlard/french-sentiment-analysis-with-bert",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@software{blard2020,
  author = {Théophile Blard},
  title = {French sentiment analysis with BERT},
  url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
  year = {2020},
}
""",
        superseded_by="MovieReviewSentimentClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("review", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )


class MovieReviewSentimentClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MovieReviewSentimentClassification.v2",
        dataset={
            "path": "mteb/movie_review_sentiment",
            "revision": "4e182033cbfe75ae0556cd640d028986be82afd8",
        },
        description="The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/TheophileBlard/french-sentiment-analysis-with-bert",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@software{blard2020,
  author = {Théophile Blard},
  title = {French sentiment analysis with BERT},
  url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
  year = {2020},
}
""",
        adapted_from=["MovieReviewSentimentClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
