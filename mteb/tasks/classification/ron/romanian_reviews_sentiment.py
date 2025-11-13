from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RomanianReviewsSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianReviewsSentiment",
        description="LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian",
        reference="https://arxiv.org/abs/2101.04197",
        dataset={
            "path": "mteb/RomanianReviewsSentiment",
            "revision": "acfa5c2974c1d6f889e0ddf86a588b59aefd5e22",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-01", "2021-01-11"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{tache2101clustering,
  author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
  journal = {ArXiv},
  title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
  year = {2021},
}
""",
        superseded_by="RomanianReviewsSentiment.v2",
    )


class RomanianReviewsSentimentV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RomanianReviewsSentiment.v2",
        description="LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/2101.04197",
        dataset={
            "path": "mteb/romanian_reviews_sentiment",
            "revision": "6b320d55fcf5fc184a9e7cc828debb34f7949432",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-01", "2021-01-11"),
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{tache2101clustering,
  author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
  journal = {ArXiv},
  title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
  year = {2021},
}
""",
        adapted_from=["RomanianReviewsSentiment"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
